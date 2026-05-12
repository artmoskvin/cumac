// Pure integer arithmetic — branchy modular ops.
// out[i] = ((a[i] * 1103515245 + b[i] + 12345) ^ (a[i] >> 3)) & 0x7fffffff;
extern "C" __global__
void int_arith(int *out, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = a[i] * 1103515245 + b[i] + 12345;
    v ^= (a[i] >> 3);
    out[i] = v & 0x7fffffff;
}
