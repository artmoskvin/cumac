// Per-block dot products of two vectors. Each block produces one partial.
extern "C" __global__
void dot_blocks(float *partials, const float *a, const float *b, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = (i < n) ? a[i] * b[i] : 0.f;
    s[tid] = v;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = s[0];
}
