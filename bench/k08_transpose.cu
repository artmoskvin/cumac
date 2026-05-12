// Coalesced matrix transpose via shared-memory tile. Pad to avoid bank
// conflicts on the column-major access into Bs.
extern "C" __global__
void transpose(float *out, const float *in, int rows, int cols) {
    const int T = 32;
    __shared__ float tile[32][33];
    int x = blockIdx.x * T + threadIdx.x;
    int y = blockIdx.y * T + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    int tx = blockIdx.y * T + threadIdx.x;
    int ty = blockIdx.x * T + threadIdx.y;
    if (tx < rows && ty < cols) out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
}
