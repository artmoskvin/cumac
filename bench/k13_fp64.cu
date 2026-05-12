// Double-precision element-wise op. Tests fp64 ALU path (limited on Ampere
// consumer; 3090 has 1:64 fp64 throughput but it still works).
extern "C" __global__
void fp64_op(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i] + a[i];
}
