#!/usr/bin/env python3
"""
test_client.py v2 — Tests VRAM weight caching in vulkan_matmul_server

Protocol v2 (32-byte header):
  magic(4) M(4) N(4) K(4) type(4) a_only(4) key_lo(4) key_hi(4)
  type=0 plain matmul (A+B → C)
  type=1 cached matmul
    a_only=0: send A+B (weight will be cached on server)
    a_only=1: send A only (weight should already be in VRAM cache)

Response: magic(4) status(4) M(4) N(4) then C data
  status=0  OK
  status=2  NEED_WEIGHT (evicted, must retry with a_only=0)
"""

import socket, struct, numpy as np, time, sys
from collections import defaultdict

MAGIC         = 0x564B4D54
STATUS_OK     = 0
STATUS_NEED_W = 2
TYPE_PLAIN    = 0
TYPE_CACHED   = 1

HOST = "127.0.0.1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8097

# ─── Client-side cache tracking ───────────────────────────────
# Maps weight_key → (K, N) for weights we believe are cached on server
_warm_keys = {}

# Monotonic key counter — avoids id(B) address-reuse collisions.
# Python may reuse the same memory address for different numpy arrays if the
# old array is freed before the new one is allocated (same shape/size).
# Using id(B) as a cache key then causes the server to see a "cache hit" for
# a completely different weight matrix → protocol desync ("bad magic").
_key_seq = 0
def _next_key():
    global _key_seq
    _key_seq += 1
    return _key_seq

def _connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s

# Persistent connection for this test session
_sock = _connect()

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

def vk_matmul_cached(M, N, K, A, B, weight_key):
    """
    Cached matmul: C[M,N] = A[M,K] × B[K,N]
    On cache hit: only sends A (28KB for M=1 gen)
    On cache miss: sends A+B, server caches B in VRAM
    """
    global _warm_keys

    a_only = 1 if weight_key in _warm_keys else 0

    key_lo = weight_key & 0xFFFFFFFF
    key_hi = (weight_key >> 32) & 0xFFFFFFFF

    hdr = struct.pack('<IIIIIIII', MAGIC, M, N, K, TYPE_CACHED, a_only, key_lo, key_hi)
    _send_all(hdr)
    _send_all(A.astype(np.float32).tobytes())
    if not a_only:
        _send_all(B.astype(np.float32).tobytes())

    # Response
    rhdr = _recv_all(16)
    magic, status, rM, rN = struct.unpack('<IIII', rhdr)
    assert magic == MAGIC

    if status == STATUS_NEED_W:
        # Server evicted our key — retry with full B
        del _warm_keys[weight_key]
        return vk_matmul_cached(M, N, K, A, B, weight_key)

    assert status == STATUS_OK, f"status={status}"

    data = _recv_all(M * N * 4)
    C = np.frombuffer(data, dtype=np.float32).reshape(M, N)

    # Mark weight as cached on server
    if not a_only:
        _warm_keys[weight_key] = (K, N)

    return C

def vk_matmul_plain(M, N, K, A, B):
    """Plain (uncached) matmul for comparison."""
    hdr = struct.pack('<IIIIIIII', MAGIC, M, N, K, TYPE_PLAIN, 0, 0, 0)
    _send_all(hdr)
    _send_all(A.astype(np.float32).tobytes())
    _send_all(B.astype(np.float32).tobytes())
    rhdr = _recv_all(16)
    magic, status, rM, rN = struct.unpack('<IIII', rhdr)
    assert magic == MAGIC and status == 0
    return np.frombuffer(_recv_all(M*N*4), dtype=np.float32).reshape(M, N)

# ─── Tests ────────────────────────────────────────────────────

print(f"=== Vulkan Matmul Server v2 Test — VRAM Cache ===")
print(f"Connecting to {HOST}:{PORT}...\n")

print("── 1. Correctness (plain vs cached) ──")
for M, N, K, label in [
    (1,    256,  256,  "tiny"),
    (1,    1024, 1024, "1K gen"),
    (128,  2048, 2048, "128 prefill"),
    (1,    2048, 7168, "MoE expert"),
]:
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    key = _next_key()

    C_np = A @ B
    C_vk = vk_matmul_cached(M, N, K, A, B, key)
    err  = np.max(np.abs(C_vk - C_np))
    print(f"  {'✓' if err < 0.01 else '✗'} [{M}×{N}×{K}] {label:20s}  maxerr={err:.2e}")

print()
print("── 2. Cache speedup — simulating 120B generation tokens ──")
print("   (Upload weight ONCE, then send only activations for each token)")
print()

shapes = [
    (7168, 7168,  "q_proj/k_proj/v_proj  (attention)"),
    (7168, 7168,  "o_proj                (attention)"),
    (7168, 2048,  "expert gate_proj      (MoE FFN)"),
    (7168, 2048,  "expert up_proj        (MoE FFN)"),
    (2048, 7168,  "expert down_proj      (MoE FFN)"),
]
N_TOKENS = 20

for (K, N, label) in shapes:
    B = np.random.randn(K, N).astype(np.float32)
    key = _next_key()   # stable unique key — no id() address-reuse collisions
    # Remove from warm cache so first call uploads (key is always fresh here)
    _warm_keys.pop(key, None)

    cold_times = []
    warm_times = []

    for tok in range(N_TOKENS):
        A = np.random.randn(1, K).astype(np.float32)  # single token
        t0 = time.time()
        C = vk_matmul_cached(1, N, K, A, B, key)
        elapsed = (time.time() - t0) * 1000
        if tok == 0:
            cold_times.append(elapsed)  # first call = upload + compute
        else:
            warm_times.append(elapsed)  # subsequent = activation only + compute

    cold_avg = sum(cold_times)
    warm_avg = sum(warm_times) / len(warm_times)
    speedup  = cold_avg / warm_avg if warm_avg > 0 else 0
    wt_mb    = K * N * 4 / 1e6

    print(f"  [{K}×{N}]  {label}")
    print(f"    Cold (upload {wt_mb:.0f}MB weight): {cold_avg:.0f}ms")
    print(f"    Warm (send 4KB activation only):   {warm_avg:.1f}ms  ← {speedup:.0f}× faster")
    print()

print()
print("── 3. Throughput: tokens/sec with cached weights ──")
# Simulate a full 120B MoE forward pass for one token
# Real 120B MoE: ~80 layers, each with 4 attn projections + 2 active experts × 3 projections
LAYERS      = 4   # subset for timing
ATTN_PROJS  = 4   # Q,K,V,O
EXPERTS     = 2   # active experts per token
EXP_PROJS   = 3   # gate, up, down

# Pre-create weight matrices (simulating model loaded in POWER8 RAM)
attn_weights = [np.random.randn(7168, 7168).astype(np.float32) for _ in range(ATTN_PROJS)]
exp_weights  = [(np.random.randn(7168, 2048).astype(np.float32),
                 np.random.randn(7168, 2048).astype(np.float32),
                 np.random.randn(2048, 7168).astype(np.float32))
                for _ in range(EXPERTS)]

# Warm up the cache (first forward pass)
print("  Warming up weight cache (first token upload)...")
t_warmup = time.time()
A_dummy = np.random.randn(1, 7168).astype(np.float32)
for layer in range(LAYERS):
    for wi, W in enumerate(attn_weights):
        key = (layer * 1000 + wi) & 0xFFFFFFFFFFFFFFFF
        _warm_keys.pop(key, None)
        vk_matmul_cached(1, 7168, 7168, A_dummy, W, key)
    for ei, (Wg, Wu, Wd) in enumerate(exp_weights):
        kg = (layer * 1000 + 100 + ei * 3) & 0xFFFFFFFFFFFFFFFF
        ku = (layer * 1000 + 101 + ei * 3) & 0xFFFFFFFFFFFFFFFF
        kd = (layer * 1000 + 102 + ei * 3) & 0xFFFFFFFFFFFFFFFF
        for k, W, K, N in [(kg,Wg,7168,2048),(ku,Wu,7168,2048),(kd,Wd,2048,7168)]:
            _warm_keys.pop(k, None)
            vk_matmul_cached(1, N, K, np.random.randn(1,K).astype(np.float32), W, k)
warmup_ms = (time.time() - t_warmup) * 1000
print(f"  Warmup done in {warmup_ms:.0f}ms\n")

# Now time N tokens with warm cache
N_TIME_TOKENS = 10
t_start = time.time()
for tok in range(N_TIME_TOKENS):
    A = np.random.randn(1, 7168).astype(np.float32)
    for layer in range(LAYERS):
        for wi, W in enumerate(attn_weights):
            key = (layer * 1000 + wi) & 0xFFFFFFFFFFFFFFFF
            vk_matmul_cached(1, 7168, 7168, A, W, key)
        for ei, (Wg, Wu, Wd) in enumerate(exp_weights):
            kg = (layer * 1000 + 100 + ei * 3) & 0xFFFFFFFFFFFFFFFF
            ku = (layer * 1000 + 101 + ei * 3) & 0xFFFFFFFFFFFFFFFF
            kd = (layer * 1000 + 102 + ei * 3) & 0xFFFFFFFFFFFFFFFF
            vk_matmul_cached(1, 2048, 7168, np.random.randn(1,7168).astype(np.float32), Wg, kg)
            vk_matmul_cached(1, 2048, 7168, np.random.randn(1,7168).astype(np.float32), Wu, ku)
            vk_matmul_cached(1, 7168, 2048, np.random.randn(1,2048).astype(np.float32), Wd, kd)

elapsed_s = time.time() - t_start
ms_per_tok = elapsed_s * 1000 / N_TIME_TOKENS
ops_per_tok = LAYERS * (ATTN_PROJS + EXPERTS * EXP_PROJS)

print(f"  {N_TIME_TOKENS} tokens × {LAYERS} layers × "
      f"({ATTN_PROJS} attn + {EXPERTS}×{EXP_PROJS} MoE) ops")
print(f"  Time: {elapsed_s*1000:.0f}ms total  =  {ms_per_tok:.1f}ms/token")
print(f"  Throughput: {1000/ms_per_tok:.2f} tokens/sec  "
      f"({ops_per_tok} matmuls/token, {ms_per_tok/ops_per_tok:.1f}ms each)")
print(f"\n  Cache stats: {len(_warm_keys)} weights resident in VRAM")
print(f"  (Extrapolated full 80-layer 120B: "
      f"~{ms_per_tok * 80/LAYERS:.0f}ms/token = "
      f"~{1000/(ms_per_tok * 80/LAYERS):.2f} t/s)")

_sock.close()
print("\nAll done.")
