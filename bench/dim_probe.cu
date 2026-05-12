// dim_probe.cu — write each of the six launch built-ins to its own slot in
// `out`. Used to inspect which `c[0x0][...]` cmem offsets nvcc emits for
// blockDim.x/y/z and gridDim.x/y/z. Run via:
//
//     cumac run stress/dim_probe.cu dim_probe \
//         out:out:i32:6 --grid 5,7,3 --block 4,2,6
//
// Expected output (head): [4, 2, 6, 5, 7, 3]
//
// Only thread 0 of block 0 writes — otherwise every thread races on the
// same six addresses. The boundary check itself reads blockIdx/threadIdx
// from special registers (S2R SR_CTAID.X etc.), not from cmem.
extern "C" __global__
void dim_probe(int *out) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;
    out[0] = blockDim.x;   // expect: c[0x0][0x0]
    out[1] = blockDim.y;   // expect: c[0x0][0x4]
    out[2] = blockDim.z;   // expect: c[0x0][0x8]
    out[3] = gridDim.x;    // expect: c[0x0][0xc]
    out[4] = gridDim.y;    // expect: c[0x0][0x10]
    out[5] = gridDim.z;    // expect: c[0x0][0x14]
}
