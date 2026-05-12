// Tiled matmul using shared memory. Block is 16x16; tile dim must match.
// N must be a multiple of 16 (no boundary handling for simplicity).
extern "C" __global__
void matmul_tiled(float *C, const float *A, const float *B, int N) {
    const int T = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < N; t += T) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}
