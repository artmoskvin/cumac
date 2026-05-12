// Sum a buffer into a single-element output via atomicAdd.
// Forces global-memory atomics on the GPU.
extern "C" __global__
void reduce_atomic(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);
}
