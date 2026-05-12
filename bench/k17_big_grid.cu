// Large grid (16M threads) — tests command-queue / launch path handling.
// Just writes index back to out[i] so we can verify a few samples.
extern "C" __global__
void big_grid(unsigned int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (unsigned)i;
}
