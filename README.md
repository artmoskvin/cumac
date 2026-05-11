# cumac

Write a CUDA kernel, run it on macOS via [tinygrad](https://github.com/tinygrad/tinygrad)'s
NV runtime on an NVIDIA eGPU.

## Requirements

- Apple Silicon Mac with an NVIDIA eGPU over Thunderbolt/USB4
- [TinyGPU.app](https://github.com/tinygrad/tinygpu_releases) installed (tinygrad fetches it on first use)
- Docker Desktop (for `nvcc`)

## Install

```fish
uv sync
```

## Usage

```fish
# Run a kernel end-to-end (compile + launch + dump output to NAME.bin)
cumac run src/cumac/vec_add.cu vec_add \
    out:c:f32:1024 in:a:f32:1024:iota in:b:f32:1024:iota val:n:i32:1024 \
    --grid 4 --block 256

# Static analysis: registers, occupancy, SASS instruction mix
cumac analyze src/cumac/vec_add.cu vec_add --block 256

# Just compile to .cubin
cumac compile src/cumac/vec_add.cu -o vec_add.cubin
```

Kernels must use `extern "C"` and put pointer args before scalar args
(tinygrad's `NVProgram` ABI). See [`src/cumac/cumac.py`](src/cumac/cumac.py)
for the full arg-spec syntax.
