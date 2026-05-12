// Block-level reduction into per-block partial sums, using shared memory and
// __syncthreads. The host can sum the per-block partials (no second kernel).
extern "C" __global__
void reduce_shared(float *partials, const float *in, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = s[0];
}
