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

// 64x64 tiling
// M_A : Load 64x16
// M_B : Load 16x64
static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
// reg tiling 4x4
static const uint TM = 4;
static const uint TN = 4;

// Preventing Bank Conflict - add 1 padding
groupshared float s_A[BK][BM + 1];
groupshared float s_B[BK][BN + 1];

[shader("compute")]
[numthreads(16, 16, 1)] // 256 Threads
void gemm(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    let tid = local_id.y * 16 + local_id.x; // 0 ~ 255

    let block_row = group_id.y * BM;
    let block_col = group_id.x * BN;

    float reg_C[TM][TN];
    [unroll] for (uint i = 0; i < TM; ++i) {
        [unroll] for (uint j = 0; j < TN; ++j) reg_C[i][j] = 0.0f;
    }

    let ptr_A4 = (float4*)g_args.A;
    let ptr_B4 = (float4*)g_args.B;

    let K_vec = g_args.K >> 2;
    let N_vec = g_args.N >> 2;
    let num_tiles = (g_args.K + BK - 1) / BK;

    for (uint t = 0; t < num_tiles; ++t) {
        // 1. Global -> Shared (Matrix A):
        let a_r = tid >> 2;                // 0 ~ 63 (row)
        let a_c_vec = tid & 3;             // 0 ~ 3 (col vec)
        let g_r_a = block_row + a_r;
        let g_c_a_vec = (t * BK >> 2) + a_c_vec;

        if (g_r_a < g_args.M && (g_c_a_vec << 2) < g_args.K) {
            let vec = ptr_A4[g_r_a * K_vec + g_c_a_vec];
            let a_c = a_c_vec << 2;
            s_A[a_c + 0][a_r] = vec.x;
            s_A[a_c + 1][a_r] = vec.y;
            s_A[a_c + 2][a_r] = vec.z;
            s_A[a_c + 3][a_r] = vec.w;  
        } else {
            let a_c = a_c_vec << 2;
            s_A[a_c + 0][a_r] = 0.0f; s_A[a_c + 1][a_r] = 0.0f;
            s_A[a_c + 2][a_r] = 0.0f; s_A[a_c + 3][a_r] = 0.0f;
        }

        // 2. Global -> Shared (Matrix B):
        let b_r = tid >> 4;                // 0 ~ 15 (row)
        let b_c_vec = tid & 15;            // 0 ~ 15 (col vec)
        let g_r_b = t * BK + b_r;
        let g_c_b_vec = (block_col >> 2) + b_c_vec;

        if (g_r_b < g_args.K && (g_c_b_vec << 2) < g_args.N) {
            let vec = ptr_B4[g_r_b * N_vec + g_c_b_vec];
            let b_c = b_c_vec << 2;
            s_B[b_r][b_c + 0] = vec.x;
            s_B[b_r][b_c + 1] = vec.y;
            s_B[b_r][b_c + 2] = vec.z;
            s_B[b_r][b_c + 3] = vec.w;
        } else {
            let b_c = b_c_vec << 2;
            s_B[b_r][b_c + 0] = 0.0f; s_B[b_r][b_c + 1] = 0.0f;
            s_B[b_r][b_c + 2] = 0.0f; s_B[b_r][b_c + 3] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // 3. Compute Engine (4x4 Register Tile)
        float reg_A[TM];
        float reg_B[TN];

        [unroll] for (uint k = 0; k < BK; ++k) {
            [unroll] for (uint i = 0; i < TM; ++i) {
                reg_A[i] = s_A[k][local_id.y * TM + i];
            }
            [unroll] for (uint j = 0; j < TN; ++j) {
                reg_B[j] = s_B[k][local_id.x * TN + j];
            }
            [unroll] for (uint i = 0; i < TM; ++i) {
                [unroll] for (uint j = 0; j < TN; ++j) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Write Back
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
