// Kernel with no pointer args, only a scalar — sanity check on the
// "all-scalar" ABI path (bufs tuple is empty).
extern "C" __global__
void no_args(int *out, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i ^ seed;
}
