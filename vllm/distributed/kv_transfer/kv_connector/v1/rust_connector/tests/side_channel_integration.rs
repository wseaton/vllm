// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! integration tests for side channel metadata exchange

use std::thread;
use std::time::Duration;
use vllm_nixl::side_channel::tcp::TcpSideChannel;
use vllm_nixl::side_channel::SideChannel;

#[test]
fn test_two_agents_metadata_exchange() {
    // simulate two vllm deployments exchanging metadata
    let port_a = 20001;
    let port_b = 20002;

    let metadata_a = b"agent_a_metadata_v1".to_vec();
    let metadata_b = b"agent_b_metadata_v2".to_vec();

    // agent A
    let mut channel_a = TcpSideChannel::new();
    channel_a
        .start_listener(port_a, metadata_a.clone())
        .expect("agent A failed to start listener");

    // agent B
    let mut channel_b = TcpSideChannel::new();
    channel_b
        .start_listener(port_b, metadata_b.clone())
        .expect("agent B failed to start listener");

    // wait for listeners to be ready
    thread::sleep(Duration::from_millis(100));

    // agent A requests metadata from agent B
    let received_b = channel_a
        .request_metadata("127.0.0.1", port_b, 5000)
        .expect("agent A failed to request from agent B");

    assert_eq!(received_b, metadata_b);

    // agent B requests metadata from agent A
    let received_a = channel_b
        .request_metadata("127.0.0.1", port_a, 5000)
        .expect("agent B failed to request from agent A");

    assert_eq!(received_a, metadata_a);

    // cleanup
    channel_a.shutdown().expect("failed to shutdown agent A");
    channel_b.shutdown().expect("failed to shutdown agent B");
}

#[test]
fn test_concurrent_metadata_requests() {
    // test multiple clients requesting metadata concurrently
    let port = 20003;
    let metadata = b"concurrent_test_metadata".to_vec();

    let mut server = TcpSideChannel::new();
    server
        .start_listener(port, metadata.clone())
        .expect("failed to start server");

    thread::sleep(Duration::from_millis(100));

    // spawn multiple client threads
    let mut handles = vec![];
    for i in 0..5 {
        let meta_clone = metadata.clone();
        let handle = thread::spawn(move || {
            let client = TcpSideChannel::new();
            let received = client
                .request_metadata("127.0.0.1", port, 5000)
                .expect(&format!("client {} failed", i));
            assert_eq!(received, meta_clone);
        });
        handles.push(handle);
    }

    // wait for all clients to complete
    for handle in handles {
        handle.join().expect("client thread panicked");
    }

    server.shutdown().expect("failed to shutdown server");
}

#[test]
fn test_large_metadata_exchange() {
    // test with larger metadata (simulating real NixlAgentMetadata with many blocks)
    let port = 20004;

    // create ~100KB metadata (similar to real vllm metadata with many blocks)
    let mut large_metadata = Vec::new();
    for i in 0_u32..10000 {
        large_metadata.extend_from_slice(&i.to_be_bytes());
    }

    let mut server = TcpSideChannel::new();
    server
        .start_listener(port, large_metadata.clone())
        .expect("failed to start server");

    thread::sleep(Duration::from_millis(100));

    let client = TcpSideChannel::new();
    let received = client
        .request_metadata("127.0.0.1", port, 5000)
        .expect("failed to request large metadata");

    assert_eq!(received.len(), large_metadata.len());
    assert_eq!(received, large_metadata);

    server.shutdown().expect("failed to shutdown");
}

#[test]
fn test_listener_restart() {
    // verify listener can be stopped and restarted
    let port = 20005;
    let metadata_v1 = b"metadata_version_1".to_vec();
    let metadata_v2 = b"metadata_version_2_updated".to_vec();

    let mut server = TcpSideChannel::new();

    // first start
    server
        .start_listener(port, metadata_v1.clone())
        .expect("failed to start first time");
    thread::sleep(Duration::from_millis(100));

    let client = TcpSideChannel::new();
    let received = client
        .request_metadata("127.0.0.1", port, 5000)
        .expect("failed first request");
    assert_eq!(received, metadata_v1);

    // shutdown
    server.shutdown().expect("failed to shutdown");
    thread::sleep(Duration::from_millis(100));

    // restart with different metadata
    server
        .start_listener(port, metadata_v2.clone())
        .expect("failed to restart");
    thread::sleep(Duration::from_millis(100));

    let received = client
        .request_metadata("127.0.0.1", port, 5000)
        .expect("failed second request");
    assert_eq!(received, metadata_v2);

    server.shutdown().expect("failed final shutdown");
}
