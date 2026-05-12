// In-place scale by 2. Tests `inout` arg semantics.
extern "C" __global__
void scale2_inplace(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.f;
}
