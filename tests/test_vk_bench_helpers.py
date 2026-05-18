import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vk_bench


class ChunkedRecvSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.requests = []

    def recv(self, n):
        self.requests.append(n)
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > n:
            self.chunks.insert(0, chunk[n:])
            return chunk[:n]
        return chunk


class PartialSendSocket:
    def __init__(self, max_send):
        self.max_send = max_send
        self.sent = bytearray()

    def send(self, data):
        n = min(self.max_send, len(data))
        if n:
            self.sent.extend(data[:n])
        return n


def test_next_key_is_monotonic(monkeypatch):
    monkeypatch.setattr(vk_bench, "_key_seq", 0)

    assert vk_bench._next_key() == 1
    assert vk_bench._next_key() == 2
    assert vk_bench._next_key() == 3


def test_parse_port_accepts_valid_ports_and_ignores_pytest_args():
    assert vk_bench._parse_port(["vk_bench.py"]) == 8097
    assert vk_bench._parse_port(["vk_bench.py", "9001"]) == 9001
    assert vk_bench._parse_port(["pytest", "tests/test_vk_bench_helpers.py"]) == 8097


def test_recv_all_reassembles_chunked_socket_data(monkeypatch):
    fake_socket = ChunkedRecvSocket([b"ab", b"cdef", b"ghi"])
    monkeypatch.setattr(vk_bench, "_sock", fake_socket)

    assert vk_bench._recv_all(7) == b"abcdefg"
    assert fake_socket.requests == [7, 5, 1]


def test_recv_all_raises_when_server_closes_early(monkeypatch):
    fake_socket = ChunkedRecvSocket([b"abc"])
    monkeypatch.setattr(vk_bench, "_sock", fake_socket)

    try:
        vk_bench._recv_all(4)
    except ConnectionError as exc:
        assert str(exc) == "Server closed"
    else:
        raise AssertionError("expected ConnectionError")


def test_send_all_retries_until_full_payload_is_sent(monkeypatch):
    fake_socket = PartialSendSocket(max_send=3)
    monkeypatch.setattr(vk_bench, "_sock", fake_socket)

    vk_bench._send_all(b"abcdefgh")

    assert bytes(fake_socket.sent) == b"abcdefgh"


def test_send_all_raises_when_socket_reports_zero_bytes(monkeypatch):
    fake_socket = PartialSendSocket(max_send=0)
    monkeypatch.setattr(vk_bench, "_sock", fake_socket)

    try:
        vk_bench._send_all(b"abc")
    except ConnectionError as exc:
        assert str(exc) == "Send failed"
    else:
        raise AssertionError("expected ConnectionError")


def test_bench_cold_skips_shapes_larger_than_staging_buffer(monkeypatch):
    monkeypatch.setattr(vk_bench, "MAX_STAGE_BYTES", 16)

    assert vk_bench.bench_cold("too large", 3, 3) == (None, None)
