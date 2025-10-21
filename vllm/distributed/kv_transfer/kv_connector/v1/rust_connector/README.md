# vLLM NIXL Rust Connector

Rust-based Python bindings for NIXL (NVIDIA Inter-process communication eXchange Library) to replace the existing Python bindings with better performance and type safety.

## Architecture

This crate provides a PyO3-based Python module (`vllm_nixl`) that wraps the `nixl-sys` Rust bindings, exposing an API compatible with the existing `nixl._api.nixl_agent` Python bindings.

### Key Components

- **NixlAgent**: Main Python class wrapping the NIXL agent
  - Agent metadata exchange for handshakes
  - Memory registration/deregistration
  - Transfer descriptor management
  - Async transfer operations
  - Notifications between agents
  - Telemetry collection

- **nixl_agent_config**: Helper function for agent configuration

## Building

### Prerequisites

1. **NIXL library** (only needed for full NIXL features):
   ```bash
   # Clone NIXL
   git clone https://github.com/ai-dynamo/nixl.git
   cd nixl

   # Build with Rust bindings
   meson setup build -Drust=true
   ninja -C build
   sudo ninja -C build install
   ```

2. **Rust toolchain** (1.70+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **maturin**:
   ```bash
   pip install maturin
   ```

### Build the Python Module

```bash
cd vllm/distributed/kv_transfer/kv_connector/v1/nixl_rust

# Build WITHOUT NIXL (side channel only, for testing)
maturin develop

# Full build WITH NIXL support
maturin develop --release --features nixl
```

## Testing Without NIXL

You can test the side channel functionality without building NIXL:

### Quick Start: Run All Tests
```bash
# Runs all tests (Rust unit, integration, example, Python e2e)
./run_tests.sh
```

### Option 1: Rust Example
```bash
# Run standalone Rust demo
cargo run --example side_channel_demo
```

Expected output:
```
=== vLLM NIXL Side Channel Demo ===

Starting Prefill instance on port 18000...
Starting Decode instance on port 18001...

=== Handshake Scenario ===
Decode instance requests metadata from Prefill instance...
✓ Received 234 bytes in 1.2ms
...
```

### Option 2: Python E2E Test
```bash
# Build Python module (no NIXL needed)
maturin develop

# Run Python test
python test_side_channel_e2e.py
```

Expected output:
```
============================================================
vLLM NIXL Side Channel E2E Test (No NIXL Required)
============================================================

=== Test: Basic Handshake ===
Starting prefill listener on port 19000
Starting decode listener on port 19001
✓ Received 345 bytes in 0.002s
✓ Test passed!
...
```

### Option 3: Rust Integration Tests
```bash
# Run integration tests
cargo test --test side_channel_integration

# Run all tests including unit tests
cargo test
```

## Usage

The module provides a drop-in replacement for `nixl._api.nixl_agent`:

```python
# Instead of:
from nixl._api import nixl_agent

# Use:
from vllm_nixl import NixlAgent as nixl_agent

# API is identical
config = nixl_agent_config(num_threads=8)
agent = nixl_agent("my_agent", config)
metadata = agent.get_agent_metadata()
agent.add_remote_agent(remote_metadata)
# ... etc
```

## Implementation Status

### ✅ Fully Implemented

- [x] Agent creation and configuration
- [x] Metadata exchange (get_agent_metadata, add_remote_agent)
- [x] Remote agent management (remove_remote_agent)
- [x] **Memory registration** (register_memory, deregister_memory)
  - ✨ Uses `NixlDescriptor` trait for safe, idiomatic Rust API
  - Proper resource cleanup via `RegistrationHandle`
- [x] Descriptor list creation (get_reg_descs, get_xfer_descs)
- [x] Transfer preparation (prep_xfer_dlist)
- [x] Transfer execution (make_prepped_xfer, transfer)
- [x] Transfer status checking (check_xfer_state)
- [x] Telemetry collection (get_xfer_telemetry)
- [x] Notifications (send_notif, get_new_notifs)
- [x] Resource cleanup (release_dlist_handle, release_xfer_handle)
- [x] Backend management (automatic backend creation from plugin params)
- [x] **Side channel for metadata exchange**
  - ✨ Trait-based design (`SideChannel` trait)
  - TCP implementation for agent handshakes
  - Replaces ZMQ dependency
  - Swappable backends (future: ZMQ, HTTP)

### 📝 TODO (Quality Improvements)

- [ ] **Error handling improvements**
  - More granular error types
  - Better error messages with context

- [ ] **Testing**
  - Unit tests for each component
  - Integration tests matching existing nixl_connector tests

- [ ] **Performance optimizations**
  - Reduce lock contention
  - Batch operations where possible

## Architecture Details

### Memory Registration using NixlDescriptor Trait

This implementation leverages NIXL's safe Rust API by implementing the `NixlDescriptor` trait:

```rust
// VllmMemoryDescriptor implements the NixlDescriptor trait
pub struct VllmMemoryDescriptor {
    ptr: usize,
    size: usize,
    dev_id: u64,
    mem_type: MemType,
}

impl MemoryRegion for VllmMemoryDescriptor {
    unsafe fn as_ptr(&self) -> *const u8 { ... }
    fn size(&self) -> usize { ... }
}

impl NixlDescriptor for VllmMemoryDescriptor {
    fn mem_type(&self) -> MemType { ... }
    fn device_id(&self) -> u64 { ... }
}
```

**Flow:**
1. `get_reg_descs()` stores descriptor metadata (addr, size, dev_id, mem_type)
2. `register_memory()` creates `VllmMemoryDescriptor` instances
3. Calls `agent.register_memory(&descriptor, opt_args)` using trait
4. Returns `RegistrationHandle` which auto-deregisters on drop

**Benefits:**
- ✅ Type-safe, idiomatic Rust
- ✅ No unsafe FFI handle management
- ✅ Automatic resource cleanup
- ✅ Compile-time verification

### Side Channel for Metadata Exchange

The side channel enables metadata exchange between vLLM deployments during NIXL agent handshakes, replacing the ZMQ-based approach with a trait-based design.

**Trait Design:**
```rust
pub trait SideChannel: Send + Sync {
    fn start_listener(&mut self, port: u16, metadata: Vec<u8>) -> Result<()>;
    fn request_metadata(&self, host: &str, port: u16, timeout_ms: u64) -> Result<Vec<u8>>;
    fn shutdown(&mut self) -> Result<()>;
    fn is_running(&self) -> bool;
}
```

**Current Implementation:**
- `TcpSideChannel`: Simple TCP-based protocol with length-prefixed messages
- Protocol: 4-byte big-endian length + payload
- Thread-safe, concurrent client support
- Configurable timeouts (default 5s)

**Python API:**
```python
from vllm_nixl import NixlAgent

agent = NixlAgent("my_agent")

# start listener for metadata requests
metadata = agent.get_agent_metadata()
agent.start_side_channel_listener(port=8000, metadata=metadata)

# request metadata from remote agent
remote_metadata = agent.request_remote_metadata(
    host="192.168.1.100",
    port=8000,
    timeout_ms=5000  # optional, defaults to 5000
)

# cleanup
agent.stop_side_channel_listener()
```

**Future Backends:**
- `ZmqSideChannel`: Drop-in replacement using ZMQ ROUTER/REQ pattern (matches Python)
- `HttpSideChannel`: REST API-based for cloud deployments

## Integration with vLLM

To use this in vLLM's `NixlConnector`:

1. Build the module: `cd vllm/distributed/kv_transfer/kv_connector/v1/nixl_rust && maturin develop --features nixl`

2. Update imports in `nixl_connector.py`:
   ```python
   try:
       from vllm_nixl import NixlAgent as nixl_agent
       from vllm_nixl import nixl_agent_config
   except ImportError:
       from nixl._api import nixl_agent, nixl_agent_config
   ```

3. Run tests:
   ```bash
   pytest tests/v1/kv_connector/unit/test_nixl_connector.py
   ```

## Performance Benefits

Expected improvements over Python bindings:

- **Lower overhead**: Direct Rust-to-C FFI vs Python-to-C
- **Better memory management**: Rust ownership model
- **Thread safety**: Compile-time guarantees
- **Type safety**: Static typing throughout
- **Reduced GIL contention**: GIL released during blocking operations

## Known Limitations

1. **Memory registration** requires completing the FFI integration (see TODO above)
2. **Feature flag**: Must build with `--features nixl` to enable NIXL support
3. **NIXL path**: Currently hardcoded to `/tmp/nixl` in Cargo.toml

## Contributing

To extend this implementation:

1. **Adding new methods**: Follow the pattern in `lib.rs` - wrap Rust calls with `#[cfg(feature = "nixl")]`
2. **Error handling**: Use `PyRuntimeError` for runtime errors, `PyValueError` for invalid inputs
3. **Testing**: Add tests in Python that mirror existing nixl tests

## References

- [NIXL Repository](https://github.com/ai-dynamo/nixl)
- [NIXL Rust Bindings](https://github.com/ai-dynamo/nixl/tree/main/src/bindings/rust)
- [PyO3 Documentation](https://pyo3.rs/)
- [vLLM KV Connector Architecture](../README.md)
