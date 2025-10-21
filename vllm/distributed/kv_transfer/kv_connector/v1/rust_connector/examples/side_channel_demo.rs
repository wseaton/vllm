// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! demonstrates side channel communication between two simulated vllm deployments
//!
//! run with: cargo run --example side_channel_demo

use std::thread;
use std::time::Duration;
use vllm_nixl::side_channel::tcp::TcpSideChannel;
use vllm_nixl::side_channel::SideChannel;

fn main() {
    println!("=== vLLM NIXL Side Channel Demo ===\n");

    // simulate two vLLM deployments: Prefill (P) and Decode (D)
    let prefill_port = 18000;
    let decode_port = 18001;

    // simulate agent metadata (in real vLLM this would be serialized NixlAgentMetadata)
    let prefill_metadata = create_mock_metadata("prefill-instance", 1000, 4);
    let decode_metadata = create_mock_metadata("decode-instance", 2000, 8);

    println!("Starting Prefill instance on port {}...", prefill_port);
    let mut prefill_channel = TcpSideChannel::new();
    prefill_channel
        .start_listener(prefill_port, prefill_metadata.clone())
        .expect("Failed to start prefill listener");

    println!("Starting Decode instance on port {}...", decode_port);
    let mut decode_channel = TcpSideChannel::new();
    decode_channel
        .start_listener(decode_port, decode_metadata.clone())
        .expect("Failed to start decode listener");

    // give servers time to start
    thread::sleep(Duration::from_millis(200));

    println!("\n=== Handshake Scenario ===");
    println!("Decode instance requests metadata from Prefill instance...");

    // D requests metadata from P (typical remote prefill scenario)
    let start = std::time::Instant::now();
    let received_prefill_meta = decode_channel
        .request_metadata("127.0.0.1", prefill_port, 5000)
        .expect("Failed to request prefill metadata");
    let elapsed = start.elapsed();

    println!("✓ Received {} bytes in {:?}", received_prefill_meta.len(), elapsed);
    assert_eq!(received_prefill_meta, prefill_metadata);

    // P can also request from D (bidirectional)
    println!("\nPrefill instance requests metadata from Decode instance...");
    let start = std::time::Instant::now();
    let received_decode_meta = prefill_channel
        .request_metadata("127.0.0.1", decode_port, 5000)
        .expect("Failed to request decode metadata");
    let elapsed = start.elapsed();

    println!("✓ Received {} bytes in {:?}", received_decode_meta.len(), elapsed);
    assert_eq!(received_decode_meta, decode_metadata);

    // demonstrate concurrent requests (simulating multiple TP ranks)
    println!("\n=== Concurrent Requests (simulating TP ranks) ===");
    let mut handles = vec![];

    for rank in 0..4 {
        let handle = thread::spawn(move || {
            let client = TcpSideChannel::new();
            let start = std::time::Instant::now();
            let metadata = client
                .request_metadata("127.0.0.1", prefill_port, 5000)
                .expect(&format!("Rank {} failed to request metadata", rank));
            let elapsed = start.elapsed();
            println!(
                "  Rank {} received {} bytes in {:?}",
                rank,
                metadata.len(),
                elapsed
            );
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    println!("\n=== Cleanup ===");
    prefill_channel
        .shutdown()
        .expect("Failed to shutdown prefill");
    decode_channel
        .shutdown()
        .expect("Failed to shutdown decode");
    println!("✓ Both instances shut down cleanly");

    println!("\n=== Demo Complete ===");
    println!("Side channel successfully exchanged metadata between deployments!");
}

/// creates mock serialized metadata similar to real NixlAgentMetadata
fn create_mock_metadata(engine_id: &str, num_blocks: usize, tp_size: usize) -> Vec<u8> {
    use std::fmt::Write;

    let mut metadata = String::new();
    writeln!(metadata, "{{").unwrap();
    writeln!(metadata, "  \"engine_id\": \"{}\",", engine_id).unwrap();
    writeln!(metadata, "  \"num_blocks\": {},", num_blocks).unwrap();
    writeln!(metadata, "  \"tp_size\": {},", tp_size).unwrap();
    writeln!(metadata, "  \"kv_cache_layout\": \"HND\",").unwrap();
    writeln!(metadata, "  \"attn_backend\": \"FlashAttention\",").unwrap();

    // simulate block addresses
    write!(metadata, "  \"block_addrs\": [").unwrap();
    for i in 0..num_blocks.min(5) {
        if i > 0 {
            write!(metadata, ", ").unwrap();
        }
        write!(metadata, "{}", 0x1000000 + i * 0x10000).unwrap();
    }
    writeln!(metadata, "],").unwrap();

    writeln!(metadata, "  \"agent_metadata\": \"<binary>\"").unwrap();
    writeln!(metadata, "}}").unwrap();

    metadata.into_bytes()
}
