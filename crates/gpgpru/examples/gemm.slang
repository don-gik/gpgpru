struct GemmArgs {
    float* A;
    float* B;
    float* C;
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
};

[vk::push_constant]
GemmArgs g_args;

static const uint BM = 128;
static const uint BN = 64;
static const uint BK = 16;
static const uint TM = 8;
static const uint TN = 4;

groupshared float s_A[BM][BK + 1];
groupshared float s_B[BK][BN + 1];

[shader("compute")]
[numthreads(16, 16, 1)] // 256 threads
void gemm(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    let tid = local_id.y * 16 + local_id.x; 
    
    let block_row = group_id.y * BM;
    let block_col = group_id.x * BN;

    float reg_C[TM][TN];
    [unroll] for (uint i = 0; i < TM; ++i) {
        [unroll] for (uint j = 0; j < TN; ++j) reg_C[i][j] = 0.0f;
    }

    let num_tiles = (g_args.K + BK - 1) / BK;
    let ptr_A4 = (float4*)g_args.A;
    let ptr_B4 = (float4*)g_args.B;

    let K_vec = g_args.K >> 2;
    let N_vec = g_args.N >> 2;

    for (uint t = 0; t < num_tiles; ++t) {
        
        [unroll] for (uint i = 0; i < 2; ++i) {
            let load_idx = tid * 2 + i;
            let a_r = load_idx >> 2;           // load_idx / 4
            let a_c = (load_idx & 3) << 2;     // (load_idx % 4) * 4

            let g_r = block_row + a_r;
            let g_c_vec = (t * BK + a_c) >> 2;

            if (g_r < g_args.M && (t * BK + a_c) < g_args.K) {
                let vec = ptr_A4[g_r * K_vec + g_c_vec];
                s_A[a_r][a_c + 0] = vec.x;
                s_A[a_r][a_c + 1] = vec.y;
                s_A[a_r][a_c + 2] = vec.z;
                s_A[a_r][a_c + 3] = vec.w;
            } else {
                s_A[a_r][a_c + 0] = 0.0f;
                s_A[a_r][a_c + 1] = 0.0f;
                s_A[a_r][a_c + 2] = 0.0f;
                s_A[a_r][a_c + 3] = 0.0f;
            }
        }

        let b_r = tid >> 4;                // tid / 16
        let b_c = (tid & 15) << 2;         // (tid % 16) * 4

        let g_r = t * BK + b_r;
        let g_c_vec = (block_col + b_c) >> 2;

        if (g_r < g_args.K && (block_col + b_c) < g_args.N) {
            let vec = ptr_B4[g_r * N_vec + g_c_vec];
            s_B[b_r][b_c + 0] = vec.x;
            s_B[b_r][b_c + 1] = vec.y;
            s_B[b_r][b_c + 2] = vec.z;
            s_B[b_r][b_c + 3] = vec.w;
        } else {
            s_B[b_r][b_c + 0] = 0.0f;
            s_B[b_r][b_c + 1] = 0.0f;
            s_B[b_r][b_c + 2] = 0.0f;
            s_B[b_r][b_c + 3] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // FMA
        [unroll] for (uint k = 0; k < BK; ++k) {
            float reg_A[TM];
            float reg_B[TN];

            [unroll] for (uint i = 0; i < TM; ++i) reg_A[i] = s_A[local_id.y * TM + i][k];
            [unroll] for (uint j = 0; j < TN; ++j) reg_B[j] = s_B[k][local_id.x * TN + j];

            [unroll] for (uint i = 0; i < TM; ++i) {
                [unroll] for (uint j = 0; j < TN; ++j) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write to RAM
    let thread_row = block_row + local_id.y * TM;
    let thread_col = block_col + local_id.x * TN;
    let ptr_C4 = (float4*)g_args.C;

    [unroll] for (uint i = 0; i < TM; ++i) {
        let r = thread_row + i;
        if (r >= g_args.M || thread_col >= g_args.N) continue;

        let c_vec_idx = r * N_vec + (thread_col >> 2);
        let val4 = float4(reg_C[i][0], reg_C[i][1], reg_C[i][2], reg_C[i][3]);

        if (g_args.beta == 0.0f) {
            ptr_C4[c_vec_idx] = g_args.alpha * val4;
        } else {
            ptr_C4[c_vec_idx] = g_args.alpha * val4 + g_args.beta * ptr_C4[c_vec_idx];
        }
    }
}
