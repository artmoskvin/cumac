// Heavy warp divergence: different threads take wildly different control
// paths. Forces serial execution within a warp.
extern "C" __global__
void branchy(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    int t = threadIdx.x & 7;
    float r;
    if (t == 0)      r = v * v;
    else if (t == 1) r = v + 1.f;
    else if (t == 2) r = v - 1.f;
    else if (t == 3) r = __expf(v);
    else if (t == 4) r = __logf(fabsf(v) + 1.f);
    else if (t == 5) { r = 0.f; for (int k = 0; k < 16; ++k) r += v * k; }
    else if (t == 6) r = fabsf(v);
    else             r = -v;
    out[i] = r;
}
