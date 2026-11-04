// -----------------------------------------------------------------------------
// 1. Fill Kernel
// -----------------------------------------------------------------------------
struct FillParams {
    float* data;
    uint data_len;
    float val;
};

[vk::push_constant]
FillParams g_fill_args;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill(uint3 dispatch_id : SV_DispatchThreadID) {
    uint idx = dispatch_id.x;
    if (idx < g_fill_args.data_len) {
        g_fill_args.data[idx] = g_fill_args.val;
    }
}
