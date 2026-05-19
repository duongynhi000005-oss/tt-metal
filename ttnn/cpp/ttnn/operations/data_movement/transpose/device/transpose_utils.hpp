// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_device_operation_types.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <cstdint>
#include <optional>
#include <span>

namespace ttnn::operations::data_movement::transpose {

// Native-sharded eligibility probe. Single-arg form: input only (pre-derivation). Two-arg form:
// also checks output side and (when both shard_specs are concrete) input/output grid match.
bool is_native_transpose_sharding(
    const TensorSpec& input_spec, const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

// Output logical+padded shapes per transpose dim. HC TILE: dim[1] = logical H (slot 1 isn't
// tile-padded), dim[2] = round_up(logical C, TILE_HEIGHT). Used by compute_output_specs and the
// native-eligibility probe so they predict the same output shape.
struct TransposedShapes {
    ttnn::Shape logical;
    ttnn::Shape padded;
};
TransposedShapes transposed_shapes(const Tensor& input_tensor, ttnn::prim::TransposeOpDim dim);

// Synthesize a concrete shard_spec when the requested sharded output config has none, so the
// device op's compute_output_specs and select_program_factory see a fully-specified config.
// Returns the passed-in config unchanged when it isn't sharded or already has a spec.
tt::tt_metal::MemoryConfig derive_effective_output_memory_config(
    const Tensor& input_tensor,
    ttnn::prim::TransposeOpDim dim,
    const tt::tt_metal::MemoryConfig& requested_output_mem_config);

// Scale shard_spec from `from_shape` to `to_shape`; nullopt when scaling isn't exact.
std::optional<tt::tt_metal::ShardSpec> adjust_shard_spec_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

tt::tt_metal::ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

// Copy RuntimeTensorShape common args for `buffer` into `dst` (size-checked).
void copy_transpose_common_runtime_args(const tt::tt_metal::Buffer& buffer, std::span<std::uint32_t> dst);

// Refresh reader/writer common args on program-cache hits.
void refresh_transpose_common_runtime_args(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer);

}  // namespace ttnn::operations::data_movement::transpose
