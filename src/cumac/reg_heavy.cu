// Hold many simultaneously-live values to force high per-thread register
// pressure. Used to make `cumac analyze` show the register-limited occupancy
// regime distinctly from the warps/SM-limited regime.
//
// Trick: load 32 independent values from distinct addresses, compute on every
// pair, and store every one back. The loads can't be folded (different
// addresses), the stores keep all 32 live to the end, and the cross-products
// in the middle prevent the compiler from streaming a few at a time.
extern "C" __global__
void reg_heavy(float *out, const float *in, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    #define R32(F) F( 0) F( 1) F( 2) F( 3) F( 4) F( 5) F( 6) F( 7) \
                  F( 8) F( 9) F(10) F(11) F(12) F(13) F(14) F(15) \
                  F(16) F(17) F(18) F(19) F(20) F(21) F(22) F(23) \
                  F(24) F(25) F(26) F(27) F(28) F(29) F(30) F(31)

    // Load 32 values from different addresses; nvcc can't coalesce these
    // into a single register because each comes from a distinct location.
    #define LOAD(k) float r##k = in[i + k * stride];
    R32(LOAD)

    // Cross-multiply: each r_k becomes a function of *all* the others, so
    // every register must remain live until the very last fmaf.
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        #define UPDATE(k) r##k = fmaf(r##k, r##k, r0 + r7 + r15 + r23 + r31);
        R32(UPDATE)
    }

    // Store every one back to a distinct address. The compiler can't drop any.
    #define STORE(k) out[i + k * stride] = r##k;
    R32(STORE)
}
