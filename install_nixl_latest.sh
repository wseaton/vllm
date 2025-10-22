#!/bin/bash
#
# Updated NIXL installation script with latest versions
# - UCX 1.19.0 (latest)
# - NIXL 0.6.1 (latest)
# - Rust bindings enabled
#

set -e  # Exit on error

FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
fi

SUDO=false
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO=true
fi
ARCH=$(uname -m)

export CUDA_HOME=/usr/local/cuda
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/$ARCH-linux-gnu:$LD_LIBRARY_PATH"

ROOT_DIR="$HOME/local"
mkdir -p "$ROOT_DIR"
GDR_HOME="$ROOT_DIR/gdrcopy"
UCX_HOME="$ROOT_DIR/ucx"
export PATH="$GDR_HOME/bin:$UCX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GDR_HOME/lib:$UCX_HOME/lib:$LD_LIBRARY_PATH"

TEMP_DIR="nixl_installer_latest"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo "Installing Python build dependencies..."
pip install meson ninja pybind11

# ============================================================================
# GDRCopy Installation
# ============================================================================
if [ ! -e "/dev/gdrdrv" ] || [ "$FORCE" = true ]; then
    echo "Installing gdrcopy v2.5..."
    wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.5.tar.gz
    tar xzf v2.5.tar.gz; rm v2.5.tar.gz
    cd gdrcopy-2.5
    make prefix=$GDR_HOME CUDA=$CUDA_HOME all install

    if $SUDO; then
        echo "Running insmod.sh with sudo"
        sudo ./insmod.sh
    else
        echo "Skipping insmod.sh - sudo not available"
        echo "Please run 'sudo ./gdrcopy-2.5/insmod.sh' manually if needed"
    fi

    cd ..
else
    echo "Found /dev/gdrdrv. Skipping gdrcopy installation"
fi

# ============================================================================
# UCX Installation (v1.19.0 - LATEST)
# ============================================================================
if ! command -v ucx_info &> /dev/null || [ "$FORCE" = true ]; then
    echo "Installing UCX v1.19.0 (latest)..."
    wget https://github.com/openucx/ucx/releases/download/v1.19.0/ucx-1.19.0.tar.gz
    tar xzf ucx-1.19.0.tar.gz; rm ucx-1.19.0.tar.gz
    cd ucx-1.19.0

    # Checking Mellanox NICs
    MLX_OPTS=""
    if lspci | grep -i mellanox > /dev/null || command -v ibstat > /dev/null; then
        echo "Mellanox NIC detected, adding Mellanox-specific options"
        MLX_OPTS="--with-rdmacm \
                  --with-mlx5-dv \
                  --with-ib-hw-tm"
    fi

    ./configure  --prefix=$UCX_HOME                \
                --enable-shared                    \
                --disable-static                   \
                --disable-doxygen-doc              \
                --enable-optimizations             \
                --enable-cma                       \
                --enable-devel-headers             \
                --with-cuda=$CUDA_HOME             \
                --with-dm                          \
                --with-gdrcopy=$GDR_HOME           \
                --with-verbs                       \
                --enable-mt                        \
                --without-go                       \
                $MLX_OPTS
    make -j$(nproc)
    make -j$(nproc) install-strip

    if $SUDO; then
        echo "Running ldconfig with sudo"
        sudo ldconfig
    else
        echo "Skipping ldconfig - sudo not available"
        echo "Please run 'sudo ldconfig' manually if needed"
    fi

    cd ..
else
    echo "Found existing UCX. Skipping UCX installation"
fi

# ============================================================================
# hwloc Installation (required by NIXL)
# ============================================================================
if ! pkg-config --exists hwloc || [ "$FORCE" = true ]; then
    echo "Installing hwloc (required by NIXL)..."
    wget https://download.open-mpi.org/release/hwloc/v2.11/hwloc-2.11.2.tar.gz
    tar xzf hwloc-2.11.2.tar.gz; rm hwloc-2.11.2.tar.gz
    cd hwloc-2.11.2
    ./configure --prefix=$HOME/.local --enable-cuda
    make -j$(nproc)
    make install
    cd ..
else
    echo "Found existing hwloc. Skipping hwloc installation"
fi

# Update PKG_CONFIG_PATH for hwloc
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$HOME/.local/lib/$ARCH-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
export CPPFLAGS="-I$HOME/.local/include $CPPFLAGS"
export LDFLAGS="-L$HOME/.local/lib -L$HOME/.local/lib/$ARCH-linux-gnu $LDFLAGS"

# ============================================================================
# NIXL Installation (v0.6.1 - LATEST with Rust bindings)
# ============================================================================
if ! command -v nixl_test &> /dev/null || [ "$FORCE" = true ]; then
    echo "Installing NIXL v0.6.1 (latest) with Rust bindings..."
    wget https://github.com/ai-dynamo/nixl/archive/refs/tags/0.6.1.tar.gz
    tar xzf 0.6.1.tar.gz; rm 0.6.1.tar.gz
    cd nixl-0.6.1

    # Configure with meson - enable Rust bindings and set paths
    CXXFLAGS="-I$HOME/.local/include" \
    LDFLAGS="-L$HOME/.local/lib -L$HOME/.local/lib/$ARCH-linux-gnu" \
    meson setup build \
        --prefix=$HOME/.local \
        -Ducx_path=$UCX_HOME \
        -Drust_bindings=enabled

    cd build
    ninja
    ninja install

    cd ../..
else
    echo "Found existing NIXL. Skipping NIXL installation"
fi

# ============================================================================
# Post-installation setup
# ============================================================================
echo ""
echo "============================================================================"
echo "Installation complete!"
echo "============================================================================"
echo ""
echo "Add these to your .bashrc or .zshrc:"
echo ""
echo "export PATH=\"\$HOME/.local/bin:$GDR_HOME/bin:$UCX_HOME/bin:\$PATH\""
echo "export LD_LIBRARY_PATH=\"\$HOME/.local/lib/$ARCH-linux-gnu:$GDR_HOME/lib:$UCX_HOME/lib:\$LD_LIBRARY_PATH\""
echo "export PKG_CONFIG_PATH=\"\$HOME/.local/lib/pkgconfig:\$HOME/.local/lib/$ARCH-linux-gnu/pkgconfig:\$PKG_CONFIG_PATH\""
echo ""
echo "NIXL Rust bindings location:"
echo "  $HOME/.local/lib/pkgconfig/nixl-sys.pc"
echo ""
echo "To test NIXL installation:"
echo "  nixl_test --help"
echo ""
echo "To verify UCX:"
echo "  ucx_info -v"
echo ""
