// 256-bin histogram via atomicAdd on int counters.
// Tests integer atomics on global memory.
extern "C" __global__
void histogram(int *bins, const unsigned char *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[in[i]], 1);
}
