// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/types/arch.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "hostdevcommon/kernel_structs.h"
#include "llk_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::sfpu_util {

const map<std::string, std::map<std::string, std::string>> sfpu_op_to_op_name = {
    // FIXME: #1157
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"silu", {{"SFPU_OP_CHAIN_0", "silu_tile_init(); silu_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
    {"sign", {{"SFPU_OP_CHAIN_0", "sign_tile_init(); sign_tile(0);"}}},
};

// Binary SFPU ops driven by `run_sfpu_binary_two_input_buffer`.
//
// Each entry maps an op name to a single kernel define:
//   SFPU_OP_CHAIN_0 — expanded once per tile inside acquire/release; contains
//                     both the init call and the op call. LHS at DST[0], RHS at
//                     DST[1], result written to DST[2] for the packer.
//
// To add a new binary SFPU op: add an entry here, add a matching arm in
// sfpu_binary_function() for the golden compute, and (if its valid input
// range differs from div) add an arm in generate_packed_sfpu_binary_inputs().
const map<std::string, std::map<std::string, std::string>> sfpu_binary_op_to_op_name = {
    {"div_binary", {{"SFPU_OP_CHAIN_0", "div_binary_tile_init(); div_binary_tile(0, 1, 2);"}}},
};

bfloat16 sfpu_function(const std::string& op_name, const bfloat16& input) {
    if (op_name == "relu") {
        return bfloat16(fmaxf(static_cast<float>(input), 0.0f));
    }
    if (op_name == "exponential") {
        return bfloat16(std::exp(static_cast<float>(input)));
    }
    if (op_name == "reciprocal") {
        return bfloat16(1 / static_cast<float>(input));
    }
    if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = static_cast<float>(input);
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return bfloat16(result);
    }
    if (op_name == "sqrt") {
        return bfloat16(sqrtf(static_cast<float>(input)));
    }
    if (op_name == "sigmoid") {
        auto x = static_cast<float>(input);
        float result = 1 / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "silu") {
        auto x = static_cast<float>(input);
        float result = x / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "log") {
        return bfloat16(logf(static_cast<float>(input)));
    }
    if (op_name == "tanh") {
        return bfloat16(std::tanh(static_cast<float>(input)));
    }
    if (op_name == "sign") {
        float val = static_cast<float>(input);
        float result = static_cast<float>((val > 0.0f) - (val < 0.0f));
        return bfloat16(result);
    }
    TT_THROW("Unsupported op_name in test");
}

// Reference implementation for binary SFPU ops.
bfloat16 sfpu_binary_function(const std::string& op_name, const bfloat16& lhs, const bfloat16& rhs) {
    if (op_name == "div_binary") {
        return bfloat16(static_cast<float>(lhs) / static_cast<float>(rhs));
    }
    TT_THROW("Unsupported binary op_name in test");
}

vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const std::string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.0001f, 4.0f, numel, seed);
    }
    if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
}

// Mirror the LLK `_prepare_div_inputs` helper: uniform in [-4, 4], then snap
// |x| < 0.25 to ±0.25 keeping the sign. Every element ends up in
// [-4, -0.25] ∪ [0.25, 4]. This exercises both halves of the sfpi `setsgn`
// path and avoids sub-normal divisors / spurious 0/0 -> NaN.
static vector<uint32_t> generate_div_operand(const unsigned int numel, const int seed) {
    auto packed = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed);
    auto unpacked = unpack_vector<bfloat16, uint32_t>(packed);
    for (auto& v : unpacked) {
        float f = static_cast<float>(v);
        float sign = (f >= 0.0f) ? 1.0f : -1.0f;
        float magnitude = std::fabs(f);
        magnitude = std::max(magnitude, 0.25f);
        v = bfloat16(sign * magnitude);
    }
    return pack_vector<uint32_t, bfloat16>(unpacked);
}

// Per-operand stimuli for binary SFPU ops. LHS and RHS use independent seeds so
// their signs and magnitudes vary independently.
std::pair<vector<uint32_t>, vector<uint32_t>> generate_packed_sfpu_binary_inputs(
    const unsigned int numel, const std::string& op_name, const int seed) {
    if (op_name == "div_binary") {
        auto lhs = generate_div_operand(numel, seed);
        auto rhs = generate_div_operand(numel, seed + 1);
        return {lhs, rhs};
    }
    TT_THROW("Unsupported binary op_name in test");
}

bool is_close_packed_sfpu_output(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    }
    if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.15f); });
    }
    if ((op_name == "exponential")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    }
    if ((op_name == "log")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.03f, 0.02f); });
    }
    return is_close_packed_vectors<bfloat16, uint32_t>(
        vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.06f, 0.006f); });
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores;
    std::string sfpu_op;
    bool approx_mode = true;
};

namespace {

// Validates that cfg describes a single-core CoreRange and returns the Quasar NodeCoord.
experimental::metal2_host_api::NodeCoord extract_single_core_node(const SfpuConfig& cfg, const char* context) {
    TT_FATAL(cfg.cores.ranges().size() == 1, "{} expects a single CoreRange (got {})", context, cfg.cores.size());
    const CoreRange& cr = *cfg.cores.ranges().begin();
    TT_FATAL(cr.start_coord == cr.end_coord, "{} expects a single-core CoreRange", context);
    return {cr.start_coord.x, cr.start_coord.y};
}

// Builds a DataflowBufferSpec with implicit sync disabled — common to all DFBs in this test.
experimental::metal2_host_api::DataflowBufferSpec make_dfb_spec(
    const char* id, const SfpuConfig& cfg, tt::DataFormat fmt) {
    return {
        .unique_id = id,
        .entry_size = static_cast<uint32_t>(cfg.tile_byte_size),
        .num_entries = static_cast<uint32_t>(cfg.num_tiles),
        .data_format_metadata = fmt,
        .disable_implicit_sync = true,
    };
}

// Converts a string→string defines map to the CompilerOptions::Defines vector form.
experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines to_kernel_defines(
    const std::map<std::string, std::string>& m) {
    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines defines;
    for (const auto& [k, v] : m) {
        defines.emplace_back(k, v);
    }
    return defines;
}

// Builds a writer_unary KernelSpec bound to a single output DFB.
experimental::metal2_host_api::KernelSpec make_writer_unary_quasar_spec(const char* kernel_id, const char* out_dfb_id) {
    return {
        .unique_id = kernel_id,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = out_dfb_id,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };
}

// Builds a reader_binary KernelSpec bound to in0 and in1 DFBs
experimental::metal2_host_api::KernelSpec make_reader_binary_quasar_spec(
    const char* kernel_id, const char* in0_dfb_id, const char* in1_dfb_id) {
    return {
        .unique_id = kernel_id,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = in0_dfb_id,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = in1_dfb_id,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .runtime_arguments_schema =
            {.named_runtime_args = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };
}

// Builds writer KernelRunParams for a single-node Quasar program.
experimental::metal2_host_api::ProgramRunParams::KernelRunParams make_writer_run_params(
    const char* kernel_id,
    const experimental::metal2_host_api::NodeCoord& node,
    uint32_t dst_addr,
    uint32_t num_tiles) {
    return {
        .kernel_spec_name = kernel_id,
        .named_runtime_args = {{
            .node = node,
            .args = {{"dst_addr", dst_addr}, {"bank_id", 0u}, {"num_tiles", num_tiles}},
        }},
    };
}

// Builds reader_binary KernelRunParams for a single-node Quasar program.
experimental::metal2_host_api::ProgramRunParams::KernelRunParams make_reader_binary_run_params(
    const char* kernel_id,
    const experimental::metal2_host_api::NodeCoord& node,
    uint32_t src0_addr,
    uint32_t src1_addr,
    uint32_t num_tiles) {
    return {
        .kernel_spec_name = kernel_id,
        .named_runtime_args = {{
            .node = node,
            .args =
                {{"src0_addr", src0_addr},
                 {"src0_bank_id", 0u},
                 {"src1_addr", src1_addr},
                 {"src1_bank_id", 0u},
                 {"num_tiles", num_tiles}},
        }},
    };
}

// Creates a writer_unary kernel on the legacy (non-Quasar) path.
tt_metal::KernelHandle create_legacy_writer_kernel(tt_metal::Program& program, const SfpuConfig& cfg) {
    return tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        cfg.cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
}

}  // namespace

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    // Host input + golden generation
    std::vector<uint32_t> packed_input = sfpu_util::generate_packed_sfpu_input(
        byte_size / sizeof(bfloat16), test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    auto input = unpack_vector<bfloat16, uint32_t>(packed_input);
    std::vector<bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_UNARY_OP"] = "1";
    sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_NEG_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1";

    if (device->arch() == ARCH::QUASAR) {
        constexpr const char* IN_DFB = "in_dfb";
        constexpr const char* OUT_DFB = "out_dfb";
        constexpr const char* READER = "reader";
        constexpr const char* WRITER = "writer";
        constexpr const char* COMPUTE = "compute";

        const auto node = extract_single_core_node(test_config, "Metal 2.0 sfpu path");

        experimental::metal2_host_api::KernelSpec reader_spec{
            .unique_id = READER,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp"},
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = IN_DFB,
                .local_accessor_name = "out",
                .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
            }},
            .runtime_arguments_schema = {.named_runtime_args = {"src_addr", "bank_id", "num_tiles"}},
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        };

        experimental::metal2_host_api::KernelSpec compute_spec{
            .unique_id = COMPUTE,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp"},
            .num_threads = 1,
            .compiler_options = {.defines = to_kernel_defines(sfpu_defines)},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN_DFB,
                     .local_accessor_name = "in",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = OUT_DFB,
                     .local_accessor_name = "out",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 }},
            .compile_time_arg_bindings =
                {{"per_core_block_cnt", static_cast<uint32_t>(test_config.num_tiles)}, {"per_core_block_dim", 1u}},
            .config_spec =
                experimental::metal2_host_api::ComputeConfiguration{
                    .math_approx_mode = test_config.approx_mode,
                },
        };

        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "sfpu_compute",
            .kernels = {reader_spec, make_writer_unary_quasar_spec(WRITER, OUT_DFB), compute_spec},
            .dataflow_buffers =
                {make_dfb_spec(IN_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format)},
            .work_units = {experimental::metal2_host_api::WorkUnitSpec{
                .unique_id = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
        };

        Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

        experimental::metal2_host_api::ProgramRunParams params;
        params.kernel_run_params = {
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
                .kernel_spec_name = READER,
                .named_runtime_args =
                    {{.node = node,
                      .args =
                          {{"src_addr", input_dram_buffer->address()},
                           {"bank_id", 0u},
                           {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
            },
            make_writer_run_params(WRITER, node, output_dram_buffer->address(), test_config.num_tiles),
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = COMPUTE},
        };
        experimental::metal2_host_api::SetProgramRunParameters(program, params);

        tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
        tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    } else {
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload;
        tt_metal::Program program = tt_metal::CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // Same runtime args for every core
        vector<uint32_t> reader_rt_args = {
            (uint32_t)input_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)test_config.num_tiles,
        };

        vector<uint32_t> writer_rt_args = {
            (uint32_t)output_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)test_config.num_tiles,
        };

        for (const CoreRange& core_range : test_config.cores.ranges()) {
            tt_metal::CircularBufferConfig l1_input_cb_config =
                tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_0, test_config.l1_input_data_format}})
                    .set_page_size(tt::CBIndex::c_0, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_input_cb_config);

            tt_metal::CircularBufferConfig l1_output_cb_config =
                tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_16, test_config.l1_output_data_format}})
                    .set_page_size(tt::CBIndex::c_16, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_output_cb_config);

            auto reader_kernel = tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/dataflow/reader_unary.cpp",
                test_config.cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto writer_kernel = create_legacy_writer_kernel(program_, test_config);

            vector<uint32_t> compute_kernel_args = {
                uint32_t(test_config.num_tiles),  // per_core_block_cnt
                1                                 // per_core_block_dim
            };

            tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                test_config.cores,
                tt_metal::ComputeConfig{
                    .math_approx_mode = test_config.approx_mode,
                    .compile_args = compute_kernel_args,
                    .defines = sfpu_defines});

            for (const CoreCoord& core_coord : core_range) {
                SetRuntimeArgs(program_, writer_kernel, core_coord, writer_rt_args);
                SetRuntimeArgs(program_, reader_kernel, core_coord, reader_rt_args);
            }
        }

        tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
    }

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

/// High-level flow:
///
///   DRAM(LHS) ─┐
///              ├─> Reader ─> in0/in1 DFB or CB c_0/c_1 ─> SFPU Compute ─> out DFB/CB c_16 ─> Writer ─> DRAM(out)
///   DRAM(RHS) ─┘
///
/// @param mesh_device Device under test.
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_binary_two_input_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    const size_t per_buffer_byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = per_buffer_byte_size,
        .page_size = per_buffer_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input0_dram_buffer = CreateBuffer(dram_config);
    auto input1_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    const uint32_t numel = per_buffer_byte_size / sizeof(bfloat16);
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [packed_lhs, packed_rhs] = sfpu_util::generate_packed_sfpu_binary_inputs(numel, test_config.sfpu_op, seed);

    auto lhs = unpack_vector<bfloat16, uint32_t>(packed_lhs);
    auto rhs = unpack_vector<bfloat16, uint32_t>(packed_rhs);
    std::vector<bfloat16> golden(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), golden.begin(), [&](const bfloat16& a, const bfloat16& b) {
        return sfpu_util::sfpu_binary_function(test_config.sfpu_op, a, b);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_binary_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_BINARY_OP"] = "1";
    // TODO(add_int PR): integer ops set SFPU_OP_BINARY_ADD_INT_INCLUDE here instead
    sfpu_defines["SFPU_OP_BINARY_DIV_INCLUDE"] = "1";

    if (device->arch() == ARCH::QUASAR) {
        constexpr const char* IN0_DFB = "in0_dfb";
        constexpr const char* IN1_DFB = "in1_dfb";
        constexpr const char* OUT_DFB = "out_dfb";
        constexpr const char* READER = "reader";
        constexpr const char* WRITER = "writer";
        constexpr const char* COMPUTE = "compute";

        const auto node = extract_single_core_node(test_config, "Metal 2.0 sfpu binary path");

        experimental::metal2_host_api::KernelSpec compute_spec{
            .unique_id = COMPUTE,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp"},
            .num_threads = 1,
            .compiler_options = {.defines = to_kernel_defines(sfpu_defines)},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN0_DFB,
                     .local_accessor_name = "in0",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN1_DFB,
                     .local_accessor_name = "in1",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = OUT_DFB,
                     .local_accessor_name = "out",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 }},
            .compile_time_arg_bindings =
                {{"per_core_block_cnt", 1u}, {"per_core_block_size", static_cast<uint32_t>(test_config.num_tiles)}},
            .config_spec =
                experimental::metal2_host_api::ComputeConfiguration{
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = test_config.approx_mode,
                },
        };

        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "sfpu_binary_compute",
            .kernels =
                {make_reader_binary_quasar_spec(READER, IN0_DFB, IN1_DFB),
                 make_writer_unary_quasar_spec(WRITER, OUT_DFB),
                 compute_spec},
            .dataflow_buffers =
                {make_dfb_spec(IN0_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(IN1_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format)},
            .work_units = {experimental::metal2_host_api::WorkUnitSpec{
                .unique_id = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
        };

        Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

        experimental::metal2_host_api::ProgramRunParams params;
        params.kernel_run_params = {
            make_reader_binary_run_params(
                READER, node, input0_dram_buffer->address(), input1_dram_buffer->address(), test_config.num_tiles),
            make_writer_run_params(WRITER, node, output_dram_buffer->address(), test_config.num_tiles),
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = COMPUTE},
        };
        experimental::metal2_host_api::SetProgramRunParameters(program, params);

        tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_lhs);
        tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_rhs);
        tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    } else {
        // Legacy WH/BH path: CB c_0/c_1/c_16 + eltwise_sfpu.cpp SFPU_BINARY_OP branch.
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload;
        tt_metal::Program program = tt_metal::CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        vector<uint32_t> reader_rt_args = {
            input0_dram_buffer->address(),
            0u,
            input1_dram_buffer->address(),
            0u,
            static_cast<uint32_t>(test_config.num_tiles),
        };
        vector<uint32_t> writer_rt_args = {
            output_dram_buffer->address(), 0u, static_cast<uint32_t>(test_config.num_tiles)};
        vector<uint32_t> compute_rt_args = {1u, static_cast<uint32_t>(test_config.num_tiles), 0u};

        for (const CoreRange& core_range : test_config.cores.ranges()) {
            tt_metal::CircularBufferConfig l1_input0_cb_config =
                tt_metal::CircularBufferConfig(
                    per_buffer_byte_size, {{tt::CBIndex::c_0, test_config.l1_input_data_format}})
                    .set_page_size(tt::CBIndex::c_0, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_input0_cb_config);

            tt_metal::CircularBufferConfig l1_input1_cb_config =
                tt_metal::CircularBufferConfig(
                    per_buffer_byte_size, {{tt::CBIndex::c_1, test_config.l1_input_data_format}})
                    .set_page_size(tt::CBIndex::c_1, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_input1_cb_config);

            tt_metal::CircularBufferConfig l1_output_cb_config =
                tt_metal::CircularBufferConfig(
                    per_buffer_byte_size, {{tt::CBIndex::c_16, test_config.l1_output_data_format}})
                    .set_page_size(tt::CBIndex::c_16, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_output_cb_config);

            auto reader_kernel = tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
                test_config.cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto writer_kernel = create_legacy_writer_kernel(program_, test_config);

            auto compute_kernel = tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                test_config.cores,
                tt_metal::ComputeConfig{
                    .math_approx_mode = test_config.approx_mode,
                    .defines = sfpu_defines,
                });

            for (const CoreCoord& core_coord : core_range) {
                SetRuntimeArgs(program_, reader_kernel, core_coord, reader_rt_args);
                SetRuntimeArgs(program_, writer_kernel, core_coord, writer_rt_args);
                SetRuntimeArgs(program_, compute_kernel, core_coord, compute_rt_args);
            }
        }

        tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_lhs);
        tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_rhs);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
    }

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu
class SingleCoreSingleMeshDeviceSfpuParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign")));

class SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "relu")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "exponential")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "log"))) {
        GTEST_SKIP();
    }
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign")));

// Binary SFPU parameterized test fixture.
//
// Each test instance is identified by (num_tiles, op_name). The op_name picks
// up macro substitutions from sfpu_binary_op_to_op_name and a host-side
// reference from sfpu_binary_function. The SFPU-binary branch of the compute
// kernel uses a fresh tile_regs_acquire/release per pair (only DST[0]/DST[1]),
// so num_tiles is not bounded by DST capacity.
class SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture, TensixSfpuBinaryCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    // TODO: integer ops use tt::DataFormat::Int32 and tt::tile_size(tt::DataFormat::Int32)
    const tt::DataFormat data_format = tt::DataFormat::Float16_b;
    const size_t tile_byte_size = 2 * 32 * 32;

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = tile_byte_size,
        .l1_input_data_format = data_format,
        .l1_output_data_format = data_format,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing binary SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_binary_two_input_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuBinaryCompute,
    SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture,
    ::testing::Values(std::make_tuple(1, "div_binary"), std::make_tuple(4, "div_binary")));

}  // namespace tt::tt_metal
