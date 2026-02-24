#!/usr/bin/env python3
"""
vk_bench.py — Vulkan Matmul Server v3 Benchmark
================================================
Sweeps batch size M from 1 → 1024 across LLM-relevant weight shapes
to find the GPU/CPU crossover point and verify the fast-path speedup.

Usage:
    python3 vk_bench.py [port]          # defaults to 8097

Output:
    Per-shape tables: M, GPU ms/op, CPU ms/op, GFLOPS, speedup
    Crossover summary: smallest M where GPU beats CPU

Shapes tested (matching 120B MoE model dimensions):
    [7168 × 7168]  — attention Q/K/V/O projections
    [7168 × 2048]  — MoE expert gate_proj / up_proj
    [2048 × 7168]  — MoE expert down_proj
"""

import socket, struct, numpy as np, time, sys, os

MAGIC         = 0x564B4D54
STATUS_OK     = 0
STATUS_NEED_W = 2
TYPE_PLAIN    = 0
TYPE_CACHED   = 1

HOST = "127.0.0.1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8097

# ─── Persistent connection ────────────────────────────────────────────────────

_sock = None

def _connect():
    global _sock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(120.0)
    _sock = s

def _recv_all(n):
    buf = b''
    while len(buf) < n:
        chunk = _sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("Server closed")
        buf += chunk
    return buf

def _send_all(data):
    mv = memoryview(data)
    sent = 0
    while sent < len(mv):
        n = _sock.send(mv[sent:])
        if n == 0:
            raise ConnectionError("Send failed")
        sent += n

# ─── Client-side warm key tracking ───────────────────────────────────────────

_warm_keys = {}   # key → (K, N)
_key_seq   = 0

def _next_key():
    global _key_seq
    _key_seq += 1
    return _key_seq

def vk_matmul_cached(M, N, K, A, B, weight_key):
    """Cached matmul.  First call uploads B; subsequent calls send A only."""
    a_only = 1 if weight_key in _warm_keys else 0

    key_lo = weight_key & 0xFFFFFFFF
    key_hi = (weight_key >> 32) & 0xFFFFFFFF

    hdr = struct.pack('<IIIIIIII', MAGIC, M, N, K, TYPE_CACHED, a_only, key_lo, key_hi)
    _send_all(hdr)
    _send_all(A.astype(np.float32).tobytes())
    if not a_only:
        _send_all(B.astype(np.float32).tobytes())

    rhdr = _recv_all(16)
    magic, status, rM, rN = struct.unpack('<IIII', rhdr)
    assert magic == MAGIC, f"bad magic 0x{magic:08X}"

    if status == STATUS_NEED_W:
        del _warm_keys[weight_key]
        return vk_matmul_cached(M, N, K, A, B, weight_key)

    assert status == STATUS_OK, f"status={status}"
    data = _recv_all(M * N * 4)

    if not a_only:
        _warm_keys[weight_key] = (K, N)

    return np.frombuffer(data, dtype=np.float32).reshape(M, N)


# ─── Benchmark helpers ────────────────────────────────────────────────────────

WARM_RUNS   = 2    # runs to warm GPU before timing
TIMED_RUNS  = 10   # timed runs per (shape, M)
CPU_SAMPLES = 5    # numpy @ runs for CPU baseline

# Maximum bytes we'll send in one go (must fit in server staging = 240MB)
# A[M,K] + B[K,N] must be < 240MB for warm-up cold call
MAX_STAGE_BYTES = 230 * 1024 * 1024

def bench_shape(label, K, N):
    """Benchmark a weight shape [K,N] across a sweep of M values."""
    print(f"\n{'─'*70}")
    print(f"  Shape [{K}×{N}]  {label}")
    print(f"{'─'*70}")
    print(f"  {'M':>6}  {'GPU ms':>8}  {'CPU ms':>8}  {'GFLOPS':>8}  {'Speedup':>8}  Path")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")

    wt_mb  = K * N * 4 / 1e6
    B      = np.random.randn(K, N).astype(np.float32)
    key    = _next_key()
    _warm_keys.pop(key, None)  # ensure cold start

    # Cold upload: push B into VRAM
    A_dummy = np.random.randn(1, K).astype(np.float32)
    cold_ok  = (1 * K * 4 + K * N * 4) <= MAX_STAGE_BYTES
    if not cold_ok:
        print(f"  SKIP — weight ({wt_mb:.0f}MB) too large for staging")
        return {}

    vk_matmul_cached(1, N, K, A_dummy, B, key)   # upload B, warm pipeline

    results = {}
    M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for M in M_values:
        szA = M * K * 4
        szC = M * N * 4
        # Skip if A alone can't fit in staging
        if szA > MAX_STAGE_BYTES:
            break

        A = np.random.randn(M, K).astype(np.float32)

        # GPU: warm runs (don't time)
        for _ in range(WARM_RUNS):
            vk_matmul_cached(M, N, K, A, B, key)

        # GPU: timed runs (warm cache, a_only=1)
        t0 = time.perf_counter()
        for _ in range(TIMED_RUNS):
            vk_matmul_cached(M, N, K, A, B, key)
        gpu_ms = (time.perf_counter() - t0) * 1000 / TIMED_RUNS

        # CPU baseline: numpy matmul
        t0 = time.perf_counter()
        for _ in range(CPU_SAMPLES):
            _ = A @ B
        cpu_ms = (time.perf_counter() - t0) * 1000 / CPU_SAMPLES

        flops   = 2.0 * M * K * N          # multiply-add pairs
        gflops  = (flops / 1e9) / (gpu_ms / 1000.0)
        speedup = cpu_ms / gpu_ms

        # Indicate which path was taken
        path = "fast" if M * K * 4 + M * N * 4 <= 230 * 1024 * 1024 else "slow"

        marker = " ←" if speedup >= 1.0 else ""
        print(f"  {M:>6}  {gpu_ms:>8.1f}  {cpu_ms:>8.2f}  "
              f"{gflops:>8.1f}  {speedup:>7.2f}x  {path}{marker}")

        results[M] = {"gpu_ms": gpu_ms, "cpu_ms": cpu_ms,
                      "gflops": gflops, "speedup": speedup}

    # Find crossover
    crossover = None
    for M in sorted(results.keys()):
        if results[M]["speedup"] >= 1.0:
            crossover = M
            break

    if crossover is not None:
        print(f"\n  GPU wins at M ≥ {crossover}  "
              f"(first speedup: {results[crossover]['speedup']:.2f}×)")
    else:
        print(f"\n  GPU slower than CPU at all tested M values")

    return results

# ─── Cold upload timing ───────────────────────────────────────────────────────

def bench_cold(label, K, N):
    """Measure cold upload cost: how long to push weight to VRAM for first time."""
    wt_mb = K * N * 4 / 1e6
    if K * N * 4 + K * 4 > MAX_STAGE_BYTES:
        return None, None

    B   = np.random.randn(K, N).astype(np.float32)
    A   = np.random.randn(1, K).astype(np.float32)
    key = _next_key()
    _warm_keys.pop(key, None)

    t0 = time.perf_counter()
    vk_matmul_cached(1, N, K, A, B, key)
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm follow-up
    t0 = time.perf_counter()
    for _ in range(TIMED_RUNS):
        vk_matmul_cached(1, N, K, A, B, key)
    warm_ms = (time.perf_counter() - t0) * 1000 / TIMED_RUNS

    return cold_ms, warm_ms


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Vulkan Matmul Server v3 Benchmark")
    print(f"  Server: {HOST}:{PORT}")
    print("=" * 70)

    _connect()
    print(f"  Connected.\n")

    # ── Section 1: Cold vs warm upload timing ──────────────────────────────
    print("── 1. Cold vs Warm Upload Times ──")
    print(f"  {'Shape':>22}  {'Weight':>8}  {'Cold ms':>9}  {'Warm ms':>9}  {'Upload speedup':>14}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*14}")

    cold_shapes = [
        (7168, 7168, "attn Q/K/V/O"),
        (7168, 2048, "MoE gate/up"),
        (2048, 7168, "MoE down"),
        (1024, 1024, "small test"),
    ]
    for K, N, lbl in cold_shapes:
        cold_ms, warm_ms = bench_cold(lbl, K, N)
        if cold_ms is not None:
            upload_sx = cold_ms / warm_ms if warm_ms > 0 else 0
            wt_mb = K * N * 4 / 1e6
            print(f"  [{K:4d}×{N:4d}] {lbl:12s}  {wt_mb:6.0f}MB  "
                  f"{cold_ms:9.0f}  {warm_ms:9.1f}  {upload_sx:13.0f}×")

    # ── Section 2: M-sweep per shape ──────────────────────────────────────
    shapes = [
        (7168, 7168, "attention Q/K/V/O"),
        (7168, 2048, "MoE gate_proj / up_proj"),
        (2048, 7168, "MoE down_proj"),
    ]
    all_results = {}
    for K, N, label in shapes:
        all_results[(K, N)] = bench_shape(label, K, N)

    # ── Section 3: Throughput summary ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("── 3. 120B MoE Token Throughput Estimate ──")
    print("   (4 attn layers × 4 projs) + (4 layers × 2 experts × 3 projs) = 40 ops/token")

    # Use M=1 warm GPU times from the 7168×7168 and 7168/2048×7168 shapes
    attn_key  = (7168, 7168)
    moe_up_k  = (7168, 2048)
    moe_dn_k  = (2048, 7168)

    def get_ms(shape, M=1):
        r = all_results.get(shape, {})
        return r.get(M, {}).get("gpu_ms", None)

    attn_ms = get_ms(attn_key)
    moe_up  = get_ms(moe_up_k)
    moe_dn  = get_ms(moe_dn_k)

    if attn_ms and moe_up and moe_dn:
        # 4 layers in benchmark; extrapolate to 80 layers
        per_layer_ms = 4 * attn_ms + 2 * (moe_up + moe_dn)
        total_80L_ms = per_layer_ms * (80 / 4)
        tps          = 1000.0 / total_80L_ms
        print(f"\n   Attn proj (M=1): {attn_ms:.1f}ms  "
              f"MoE gate/up: {moe_up:.1f}ms  MoE down: {moe_dn:.1f}ms")
        print(f"   Per 4-layer block: {per_layer_ms:.0f}ms")
        print(f"   Extrapolated 80-layer: {total_80L_ms:.0f}ms/token = "
              f"{tps:.3f} tokens/sec\n")
    else:
        print("   (Could not compute — some shapes skipped)\n")

    print("── 4. GPU Dispatch Floor ──")
    if attn_ms:
        print(f"   Warm M=1 [7168×7168]: {attn_ms:.1f}ms/op  "
              f"(GPU floor — dominated by fence-wait round-trip)")
        print(f"   Two-submit baseline was ~43ms; single-submit target ~22ms")
        reduction = (43.0 - attn_ms) / 43.0 * 100
        if reduction > 0:
            print(f"   Actual reduction: {attn_ms:.1f}ms  "
                  f"({reduction:.0f}% faster than v2)")
        else:
            print(f"   (v3 fast path may not be active for this shape)")

    _sock.close()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
