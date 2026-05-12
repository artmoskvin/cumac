// Element-wise multiply, simplest possible kernel beyond add.
extern "C" __global__
void vec_mul(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
