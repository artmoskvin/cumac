// Non-unit-stride copy. With stride>1, threads in a warp hit non-contiguous
// addresses → uncoalesced loads, lower throughput. Stress test for memory.
extern "C" __global__
void strided_copy(float *out, const float *in, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i * stride % n];
}
