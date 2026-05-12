// Per-row softmax over an (N, D=1024) matrix. One block per row.
// Block is 256 threads, each handles D/256 = 4 cols. Static shared mem.
extern "C" __global__
void softmax_row(float *out, const float *in, int N) {
    const int D = 1024;
    __shared__ float s[256];
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    const float *row_in  = in + row * D;
    float *row_out       = out + row * D;

    float m = -3.4e38f;
    for (int j = tid; j < D; j += blockDim.x) m = fmaxf(m, row_in[j]);
    s[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }
    float row_max = s[0];

    float sumv = 0.f;
    for (int j = tid; j < D; j += blockDim.x) sumv += __expf(row_in[j] - row_max);
    s[tid] = sumv;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    float row_sum = s[0];

    for (int j = tid; j < D; j += blockDim.x)
        row_out[j] = __expf(row_in[j] - row_max) / row_sum;
}
