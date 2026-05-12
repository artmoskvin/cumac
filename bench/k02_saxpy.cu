// Classic SAXPY: y = alpha*x + y. alpha passed as i32-bit-pattern (cumac only
// supports int scalars) then bitcast to float — same trick CUDA itself uses.
extern "C" __global__
void saxpy(float *y, const float *x, int n, int alpha_bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float alpha = __int_as_float(alpha_bits);
    y[i] = alpha * x[i] + y[i];
}
