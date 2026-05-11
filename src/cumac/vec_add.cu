// vec_add.cu — minimal example kernel for cumac
//
// Two constraints to remember:
//   1. extern "C"   → kernel name is "vec_add", not a mangled C++ symbol.
//   2. Pointers before scalars → tinygrad's NVProgram passes bufs as
//      positional and vals as a separate tuple of ints, so the cubin's
//      param ABI must put all pointer args first.

extern "C" __global__
void vec_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Run with:
//   cumac run vec_add.cu vec_add \
//       out:c:f32:1024 in:a:f32:1024:iota in:b:f32:1024:iota val:n:i32:1024 \
//       --grid 4 --block 256
//
// Expected head of c: [0, 2, 4, 6, 8, 10, 12, 14]
