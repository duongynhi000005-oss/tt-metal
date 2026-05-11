// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

void kernel_main() {
#ifdef SFPU_BINARY_OP

    // ===== Binary SFPU path =====
#ifdef ARCH_QUASAR
    uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    uint32_t per_core_block_size = get_arg(args::per_core_block_size);
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in0_id = dfb_in0.get_id();
    const uint32_t in1_id = dfb_in1.get_id();
    const uint32_t out_id = dfb_out.get_id();
#else
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);
    constexpr auto in0_id = tt::CBIndex::c_0;
    constexpr auto in1_id = tt::CBIndex::c_1;
    constexpr auto out_id = tt::CBIndex::c_16;
#endif

    init_sfpu(in0_id, out_id);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
#ifdef ARCH_QUASAR
            dfb_in0.wait_front(1);
            dfb_in1.wait_front(1);
            dfb_out.reserve_back(1);

            tile_regs_acquire();

            // Unpack values
            copy_tile_to_dst_init_short(in0_id);
            copy_tile(in0_id, /*tile_index=*/0, /*dst_index=*/0);
            copy_tile_to_dst_init_short(in1_id);
            copy_tile(in1_id, /*tile_index=*/0, /*dst_index=*/1);

            // Perform SFPU operation
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();

            // Pack results
            tile_regs_wait();
            pack_tile(/*dst_index=*/2, out_id);
            tile_regs_release();

            dfb_out.push_back(1);
            dfb_in0.pop_front(1);
            dfb_in1.pop_front(1);
#else
            cb_wait_front(in0_id, 1);
            cb_wait_front(in1_id, 1);
            cb_reserve_back(out_id, 1);

            tile_regs_acquire();
            copy_tile(in0_id, /*tile_index=*/0, /*dst_index=*/0);
            copy_tile(in1_id, /*tile_index=*/0, /*dst_index=*/1);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(/*dst_index=*/2, out_id);
            tile_regs_release();

            cb_push_back(out_id, 1);
            cb_pop_front(in0_id, 1);
            cb_pop_front(in1_id, 1);
#endif
        }
    }

#endif  // SFPU_BINARY_OP

#ifdef SFPU_UNARY_OP

    // ===== Unary SFPU path =====
#ifdef ARCH_QUASAR
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);
    DataflowBuffer buff_in(dfb::in);
    DataflowBuffer buff_out(dfb::out);
    const uint32_t in_id = buff_in.get_id();
    const uint32_t out_id = buff_out.get_id();
#else
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    CircularBuffer buff_in(tt::CBIndex::c_0);
    CircularBuffer buff_out(tt::CBIndex::c_16);
    const uint32_t in_id = tt::CBIndex::c_0;
    const uint32_t out_id = tt::CBIndex::c_16;
#endif
    init_sfpu(in_id, out_id);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        buff_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            buff_in.wait_front(1);
            copy_tile(in_id, 0, 0);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_id);
            buff_in.pop_front(1);
            tile_regs_release();
        }
        buff_out.push_back(per_core_block_dim);
    }

#endif  // SFPU_UNARY_OP
}
