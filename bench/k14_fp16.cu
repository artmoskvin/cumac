// Half-precision arithmetic via __half intrinsics. 3090 has fast fp16
// (HFMA2 vectorised 2x). Buffers are __half.
#include <cuda_fp16.h>

extern "C" __global__
void fp16_op(__half *out, const __half *a, const __half *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(__hmul(a[i], b[i]), a[i]);
}
