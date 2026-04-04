//! Metal Shading Language source strings.
//!
//! Each shader is a standalone compute kernel. Compiled at runtime
//! via `device.new_library_with_source()`.

/// All Metal shaders concatenated. Includes:
/// - `sgemm`: tiled f32 matmul C = A × B
/// - `sgemm_transb`: tiled f32 matmul C = A × B^T
/// - `q4_matvec`: fused Q4_0 × Q8_0 matrix-vector multiply
/// - `q4_vecmat`: fused Q4_0 vector-matrix multiply (scatter-accumulate)
pub const ALL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ── f16 decode helper (used by Q4 shaders) ──

static inline float decode_f16_metal(ushort bits) {
    uint sign = uint(bits & 0x8000) << 16;
    uint exp = (bits >> 10) & 0x1F;
    uint mant = bits & 0x3FF;
    if (exp == 0) return as_type<float>(sign);
    exp = exp + 127 - 15;
    return as_type<float>(sign | (exp << 23) | (mant << 13));
}

// ── f32 tiled matmul ──

constant uint TS = 32;

kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];
    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0f;
    uint tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < tiles; t++) {
        uint ac = t * TS + tid.x;
        uint br = t * TS + tid.y;
        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (br < K && col < N) ? B[br * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TS; i++) acc = fma(As[tid.y][i], Bs[i][tid.x], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

kernel void sgemm_transb(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];
    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0f;
    uint tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < tiles; t++) {
        uint ac = t * TS + tid.x;
        uint bk = t * TS + tid.y;
        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TS; i++) acc = fma(As[tid.y][i], Bs[i][tid.x], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ��─ Q4_0 fused matvec: scores[N] = Q4[N, K] @ Q8_x[K] ──

kernel void q4_matvec(
    device const uchar* Q4    [[buffer(0)]],
    device const char*  Q8    [[buffer(1)]],
    device const float* Q8s   [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    constant uint&      K     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    uint blocks = K / 32;
    uint bytes_per_row = blocks * 18;
    device const uchar* row = Q4 + tid * bytes_per_row;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar* block = row + b * 18;
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        float combined_scale = decode_f16_metal(scale_bits) * Q8s[b];
        device const uchar* quants = block + 2;
        device const char* q8 = Q8 + b * 32;
        int isum = 0;
        for (uint j = 0; j < 16; j++) {
            uchar byte = quants[j];
            char lo = char(byte & 0x0F) - 8;
            char hi = char(byte >> 4) - 8;
            isum += int(lo) * int(q8[j * 2]);
            isum += int(hi) * int(q8[j * 2 + 1]);
        }
        acc += float(isum) * combined_scale;
    }
    out[tid] = acc;
}

// ── Q4_0 fused vecmat: out[K] = activation[N] @ Q4[N, K] ──

kernel void q4_vecmat(
    device const float* activation [[buffer(0)]],
    device const uchar* Q4         [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    constant uint&      N          [[buffer(3)]],
    constant uint&      K          [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K) return;
    uint blocks_per_row = K / 32;
    uint bytes_per_row = blocks_per_row * 18;
    uint block_idx = tid / 32;
    uint elem_in_block = tid % 32;
    uint nibble_idx = elem_in_block / 2;
    bool is_high = (elem_in_block & 1) != 0;
    float acc = 0.0f;
    for (uint row = 0; row < N; row++) {
        float act = activation[row];
        if (act < 1e-10f && act > -1e-10f) continue;
        device const uchar* block = Q4 + row * bytes_per_row + block_idx * 18;
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        float q4_scale = decode_f16_metal(scale_bits);
        uchar byte = block[2 + nibble_idx];
        int q_val = is_high ? (int(byte >> 4) - 8) : (int(byte & 0x0F) - 8);
        acc += float(q_val) * q4_scale * act;
    }
    out[tid] = acc;
}
"#;
