// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Chunked-prefill helpers consumed by the compute kernels (compute_common.hpp,
// compute_streaming.hpp). Kept separate from ring_utils.hpp so the compute
// headers don't pull in RingIdSequencer / find_last_active_ring_iter — the
// experimental sibling kernel (exp_ring_joint_sdpa) defines its own copies of
// those in exp_ring_utils.hpp, and including ring_utils.hpp from the compute
// headers would collide with that copy.

#pragma once

#include <cstdint>

/**
 * Per-call chunked-prefill runtime state for sdpa_ring_v2. The compile-time per-chunk
 * geometry (q_local_padded_Nt / chunk_size_t) lives in template params; this struct
 * carries the per-chunk runtime offsets.
 */
struct ChunkedContext {
    uint32_t q_start_idx_t = 0;  // absolute Q-tile offset of this chunk's Q slab
    uint32_t ring_index = 0;     // logical ring rotation index for absolute-Q-tile compute
    bool kv_pad_rotation_enabled = false;
    uint32_t kv_actual_nt = 0;   // valid KV tiles before this fixed-size chunk
    uint32_t new_actual_nt = 0;  // valid tiles in this fixed-size chunk
    uint32_t ring_size = 0;
    uint32_t q_local_padded_Nt = 0;  // chunk_size_local in tiles
};

/**
 * Map a device-local K tile index to its global attention K position. Used by the
 * logical_n skip predicate and the diagonal-stamp mask coords. Under chunked-prefill
 * the local cache packs the per-chunk K region for each chunk back-to-back; each
 * region is q_local_padded_Nt tiles (= Q's per-device extent, since one chunk's Q
 * is one such region), so the mapping is non-monotonic.
 */
template <
    bool chunked_enabled,
    uint32_t kv_local_padded_Nt = 0,
    uint32_t chunk_size_t = 0,
    uint32_t q_local_padded_Nt = 0>
inline uint32_t kv_global_tile_for_local(uint32_t ring_id, uint32_t local_tile_idx) {
    if constexpr (chunked_enabled) {
        return (local_tile_idx / q_local_padded_Nt) * chunk_size_t + ring_id * q_local_padded_Nt +
               (local_tile_idx % q_local_padded_Nt);
    } else {
        return ring_id * kv_local_padded_Nt + local_tile_idx;
    }
}

constexpr uint32_t KV_PAD_ROTATION_INVALID_TILE = 0xFFFFFFFFu;

inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

inline uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

inline uint32_t kv_pad_old_capacity_nt(uint32_t ring_size, uint32_t chunk_size_local_nt) {
    return ring_size * chunk_size_local_nt;
}

inline uint32_t kv_pad_old_fill_nt(
    uint32_t kv_actual_nt, uint32_t new_actual_nt, uint32_t ring_size, uint32_t chunk_size_local_nt) {
    const uint32_t old_capacity_nt = kv_pad_old_capacity_nt(ring_size, chunk_size_local_nt);
    const uint32_t old_pad_nt = old_capacity_nt > kv_actual_nt ? old_capacity_nt - kv_actual_nt : 0;
    return min_u32(old_pad_nt, new_actual_nt);
}

inline uint32_t kv_pad_fill_on_chip_nt(
    uint32_t kv_actual_nt, uint32_t new_actual_nt, uint32_t ring_size, uint32_t chunk_size_local_nt, uint32_t chip) {
    const uint32_t fill_nt = kv_pad_old_fill_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt);
    const uint32_t fill_end_nt = kv_actual_nt + fill_nt;
    const uint32_t chip_start_nt = chip * chunk_size_local_nt;
    const uint32_t chip_end_nt = chip_start_nt + chunk_size_local_nt;
    const uint32_t start_nt = max_u32(kv_actual_nt, chip_start_nt);
    const uint32_t end_nt = min_u32(fill_end_nt, chip_end_nt);
    return end_nt > start_nt ? end_nt - start_nt : 0;
}

inline uint32_t kv_pad_phase2_remaining_nt(
    uint32_t kv_actual_nt, uint32_t new_actual_nt, uint32_t ring_size, uint32_t chunk_size_local_nt) {
    const uint32_t fill_nt = kv_pad_old_fill_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt);
    return new_actual_nt - fill_nt;
}

inline uint32_t kv_pad_phase2_prefix_before_chip_nt(
    uint32_t kv_actual_nt, uint32_t new_actual_nt, uint32_t ring_size, uint32_t chunk_size_local_nt, uint32_t chip) {
    uint32_t remaining_nt = kv_pad_phase2_remaining_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt);
    uint32_t prefix_nt = 0;
    for (uint32_t c = 0; c < chip; ++c) {
        const uint32_t pad_fill_nt =
            kv_pad_fill_on_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, c);
        const uint32_t avail_nt = chunk_size_local_nt - pad_fill_nt;
        const uint32_t take_nt = min_u32(avail_nt, remaining_nt);
        prefix_nt += take_nt;
        remaining_nt -= take_nt;
    }
    return prefix_nt;
}

inline uint32_t kv_pad_phase2_take_on_chip_nt(
    uint32_t kv_actual_nt, uint32_t new_actual_nt, uint32_t ring_size, uint32_t chunk_size_local_nt, uint32_t chip) {
    const uint32_t remaining_nt =
        kv_pad_phase2_remaining_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt);
    const uint32_t prefix_nt =
        kv_pad_phase2_prefix_before_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
    if (prefix_nt >= remaining_nt) {
        return 0;
    }
    const uint32_t pad_fill_nt =
        kv_pad_fill_on_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
    const uint32_t avail_nt = chunk_size_local_nt - pad_fill_nt;
    return min_u32(avail_nt, remaining_nt - prefix_nt);
}

inline uint32_t kv_pad_q_global_tile_for_local(
    uint32_t chip,
    uint32_t q_tile,
    uint32_t kv_actual_nt,
    uint32_t new_actual_nt,
    uint32_t ring_size,
    uint32_t chunk_size_local_nt) {
    const uint32_t pad_fill_nt =
        kv_pad_fill_on_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
    if (q_tile < pad_fill_nt) {
        const uint32_t chip_start_nt = chip * chunk_size_local_nt;
        return max_u32(kv_actual_nt, chip_start_nt) + q_tile;
    }

    const uint32_t phase2_row = q_tile - pad_fill_nt;
    const uint32_t phase2_take_nt =
        kv_pad_phase2_take_on_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
    if (phase2_row < phase2_take_nt) {
        const uint32_t fill_nt = kv_pad_old_fill_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt);
        const uint32_t prefix_nt =
            kv_pad_phase2_prefix_before_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
        return kv_actual_nt + fill_nt + prefix_nt + phase2_row;
    }
    return KV_PAD_ROTATION_INVALID_TILE;
}

inline uint32_t kv_pad_k_global_tile_for_local(
    uint32_t chip,
    uint32_t local_tile,
    uint32_t kv_actual_nt,
    uint32_t new_actual_nt,
    uint32_t ring_size,
    uint32_t chunk_size_local_nt) {
    const uint32_t fill_nt = kv_pad_old_fill_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt);
    if (local_tile < chunk_size_local_nt) {
        const uint32_t old_global_nt = chip * chunk_size_local_nt + local_tile;
        return old_global_nt < kv_actual_nt + fill_nt ? old_global_nt : KV_PAD_ROTATION_INVALID_TILE;
    }

    const uint32_t new_cell = local_tile - chunk_size_local_nt;
    const uint32_t phase2_take_nt =
        kv_pad_phase2_take_on_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
    if (new_cell < phase2_take_nt) {
        const uint32_t prefix_nt =
            kv_pad_phase2_prefix_before_chip_nt(kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt, chip);
        return kv_actual_nt + fill_nt + prefix_nt + new_cell;
    }
    return KV_PAD_ROTATION_INVALID_TILE;
}

inline bool kv_pad_k_chunk_has_valid_cols(
    uint32_t chip,
    uint32_t local_tile_start,
    uint32_t num_tiles,
    uint32_t kv_actual_nt,
    uint32_t new_actual_nt,
    uint32_t ring_size,
    uint32_t chunk_size_local_nt) {
    for (uint32_t col = 0; col < num_tiles; ++col) {
        if (kv_pad_k_global_tile_for_local(
                chip, local_tile_start + col, kv_actual_nt, new_actual_nt, ring_size, chunk_size_local_nt) !=
            KV_PAD_ROTATION_INVALID_TILE) {
            return true;
        }
    }
    return false;
}
