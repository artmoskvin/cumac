// 5-point stencil on a 2D grid (cross). Exercises 2D boundary handling.
// out[y,x] = 0.2 * (in[y,x] + in[y±1, x] + in[y, x±1]). Edges left untouched.
extern "C" __global__
void stencil5(float *out, const float *in, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= cols - 1 || y <= 0 || y >= rows - 1) return;
    int p = y * cols + x;
    out[p] = 0.2f * (in[p] + in[p - 1] + in[p + 1] + in[p - cols] + in[p + cols]);
}
