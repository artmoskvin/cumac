// Warp-wide reduction using __shfl_down_sync. Tests warp primitives.
// Each warp writes its 32-element sum to out[warp_id].
extern "C" __global__
void warp_reduce(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    float v = (tid < n) ? in[tid] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) out[tid >> 5] = v;
}
