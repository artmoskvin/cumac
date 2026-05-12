// Square matmul (row-major): C = A * B, all NxN. No shared-mem tiling.
// Grid is 2D — exercises blockIdx.y / threadIdx.y plumbing.
extern "C" __global__
void matmul_naive(float *C, const float *A, const float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}
