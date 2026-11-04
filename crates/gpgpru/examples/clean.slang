struct PushConstants {
    float* data;
    uint data_len;
    float factor;
};

[vk::push_constant]
PushConstants g_args;

[shader("compute")]
[numthreads(256, 1, 1)]
void scale(uint3 _sv_thread_id : SV_DispatchThreadID) {
    let idx = _sv_thread_id.x;
    if ((idx < g_args.data_len)) {
        g_args.data[idx] *= g_args.factor;
    }
}
