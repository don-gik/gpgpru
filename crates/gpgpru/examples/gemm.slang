static const uint TILE_SIZE = 16;
static const uint3 BLOCKDIM = uint3(TILE_SIZE, TILE_SIZE, 1);

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void gemm(
    uint3 thread_idx: SV_GroupThreadID,
    uint3 block_idx: SV_GroupID,
    
    uniform float* A,
    uniform float* B,
    uniform float* C,
    uniform uint M,
    uniform uint N,
    uniform uint K,
    uniform float alpha,
    uniform float beta,
) {
    const uint x = block_idx.x * BLOCKDIM.x + thread_idx.x;
    const uint y = block_idx.y * BLOCKDIM.y + thread_idx.y;

    if (x < M && y < N) {
        float tmp = 0.0;

        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
