// -----------------------------------------------------------------------------
// 2. Multiply Copy Kernel
// -----------------------------------------------------------------------------
struct MultiplyCopyParams {
    float* src;
    uint src_len;
    float* dst;
    uint dst_len;
    float multiplier;
};

[vk::push_constant]
MultiplyCopyParams g_mult_args;

[shader("compute")]
[numthreads(256, 1, 1)]
void multiply_copy(uint3 dispatch_id : SV_DispatchThreadID) {
    uint idx = dispatch_id.x;
    if (idx < g_mult_args.src_len && idx < g_mult_args.dst_len) {
        g_mult_args.dst[idx] = g_mult_args.src[idx] * g_mult_args.multiplier;
    }
}
