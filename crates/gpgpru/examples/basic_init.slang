struct PushConstants {
    float* _gpu_data_ptr;
    uint _gpu_data_len;
    float val;
};

[vk::push_constant]
PushConstants g_args;

[shader("compute")]
[numthreads(256, 1, 1)]
void scale(uint3 _sv_thread_id : SV_DispatchThreadID) {
    let idx = _sv_thread_id.x;
    if ((idx < g_args._gpu_data_len)) {
        for (int i = 0; i < 100000; i++) {
            var v = g_args._gpu_data_ptr[idx];
            var multiplier = g_args.val;
            
            v = (v * multiplier) + 0.0001f;

            g_args._gpu_data_ptr[idx] = v;
        }
    }
}
