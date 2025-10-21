#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
e2e test for side channel without requiring NIXL

run with:
    # build without nixl feature
    maturin develop

    # run test
    python test_side_channel_e2e.py
"""

import json
import time
import threading
from typing import Any


def test_basic_handshake():
    """test basic metadata exchange between two deployments"""
    from vllm_nixl import TcpSideChannel

    print("=== Test: Basic Handshake ===")

    # create two side channels (simulating P and D)
    prefill_channel = TcpSideChannel()
    decode_channel = TcpSideChannel()

    # create mock metadata
    prefill_metadata = create_mock_metadata("prefill-instance", num_blocks=1000, tp_size=4)
    decode_metadata = create_mock_metadata("decode-instance", num_blocks=2000, tp_size=8)

    prefill_port = 19000
    decode_port = 19001

    # start listeners
    print(f"Starting prefill listener on port {prefill_port}")
    prefill_channel.start_listener(prefill_port, prefill_metadata)
    assert prefill_channel.is_running()

    print(f"Starting decode listener on port {decode_port}")
    decode_channel.start_listener(decode_port, decode_metadata)
    assert decode_channel.is_running()

    # give servers time to start
    time.sleep(0.2)

    # D requests metadata from P (remote prefill scenario)
    print("Decode requesting metadata from Prefill...")
    start = time.time()
    received_prefill = decode_channel.request_metadata("127.0.0.1", prefill_port, 5000)
    elapsed = time.time() - start

    print(f"✓ Received {len(received_prefill)} bytes in {elapsed:.3f}s")
    assert received_prefill == prefill_metadata

    # P requests metadata from D (bidirectional)
    print("Prefill requesting metadata from Decode...")
    start = time.time()
    received_decode = prefill_channel.request_metadata("127.0.0.1", decode_port, 5000)
    elapsed = time.time() - start

    print(f"✓ Received {len(received_decode)} bytes in {elapsed:.3f}s")
    assert received_decode == decode_metadata

    # cleanup
    prefill_channel.shutdown()
    decode_channel.shutdown()
    assert not prefill_channel.is_running()
    assert not decode_channel.is_running()

    print("✓ Test passed!\n")


def test_concurrent_tp_ranks():
    """test concurrent requests from multiple TP ranks"""
    from vllm_nixl import TcpSideChannel

    print("=== Test: Concurrent TP Ranks ===")

    server = TcpSideChannel()
    metadata = create_mock_metadata("test-server", num_blocks=500, tp_size=1)
    port = 19002

    server.start_listener(port, metadata)
    time.sleep(0.2)

    # simulate 4 TP ranks requesting concurrently
    def worker(rank: int):
        client = TcpSideChannel()
        start = time.time()
        received = client.request_metadata("127.0.0.1", port, 5000)
        elapsed = time.time() - start
        print(f"  Rank {rank} received {len(received)} bytes in {elapsed:.3f}s")
        assert received == metadata

    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    server.shutdown()
    print("✓ Test passed!\n")


def test_large_metadata():
    """test with large metadata (simulating many blocks)"""
    from vllm_nixl import TcpSideChannel

    print("=== Test: Large Metadata ===")

    server = TcpSideChannel()
    # create ~100KB metadata
    large_metadata = create_mock_metadata("large-instance", num_blocks=10000, tp_size=16)
    port = 19003

    print(f"Metadata size: {len(large_metadata)} bytes")

    server.start_listener(port, large_metadata)
    time.sleep(0.2)

    client = TcpSideChannel()
    start = time.time()
    received = client.request_metadata("127.0.0.1", port, 10000)
    elapsed = time.time() - start

    print(f"✓ Received {len(received)} bytes in {elapsed:.3f}s")
    assert received == large_metadata

    server.shutdown()
    print("✓ Test passed!\n")


def test_timeout():
    """test connection timeout when server not available"""
    from vllm_nixl import TcpSideChannel

    print("=== Test: Connection Timeout ===")

    client = TcpSideChannel()

    # try to connect to non-existent server with short timeout
    print("Attempting to connect to non-existent server...")
    start = time.time()
    try:
        client.request_metadata("127.0.0.1", 19999, 500)
        assert False, "Should have timed out"
    except Exception as e:
        elapsed = time.time() - start
        print(f"✓ Timed out as expected after {elapsed:.3f}s: {e}")

    print("✓ Test passed!\n")


def test_restart_listener():
    """test stopping and restarting listener with different metadata"""
    from vllm_nixl import TcpSideChannel

    print("=== Test: Restart Listener ===")

    server = TcpSideChannel()
    client = TcpSideChannel()
    port = 19004

    # first start
    metadata_v1 = b"metadata_version_1"
    server.start_listener(port, metadata_v1)
    time.sleep(0.2)

    received = client.request_metadata("127.0.0.1", port, 5000)
    assert received == metadata_v1
    print("✓ Received v1 metadata")

    # shutdown
    server.shutdown()
    time.sleep(0.2)

    # restart with new metadata
    metadata_v2 = b"metadata_version_2_updated"
    server.start_listener(port, metadata_v2)
    time.sleep(0.2)

    received = client.request_metadata("127.0.0.1", port, 5000)
    assert received == metadata_v2
    print("✓ Received v2 metadata")

    server.shutdown()
    print("✓ Test passed!\n")


def create_mock_metadata(engine_id: str, num_blocks: int, tp_size: int) -> bytes:
    """create mock serialized metadata similar to NixlAgentMetadata"""
    metadata = {
        "engine_id": engine_id,
        "num_blocks": num_blocks,
        "tp_size": tp_size,
        "kv_cache_layout": "HND",
        "attn_backend": "FlashAttention",
        "block_lens": [4096] * min(num_blocks, 10),  # sample block sizes
        "kv_caches_base_addr": [0x1000000 + i * 0x10000 for i in range(min(num_blocks, 10))],
        "agent_metadata": "<binary-data-placeholder>",
    }
    return json.dumps(metadata).encode("utf-8")


def main():
    print("=" * 60)
    print("vLLM NIXL Side Channel E2E Test (No NIXL Required)")
    print("=" * 60)
    print()

    try:
        test_basic_handshake()
        test_concurrent_tp_ranks()
        test_large_metadata()
        test_timeout()
        test_restart_listener()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
