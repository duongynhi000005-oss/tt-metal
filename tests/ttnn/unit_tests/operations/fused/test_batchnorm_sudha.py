import torch
import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import comp_relative_frobenius

# 1. Setup device
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

torch_arg1 = torch.load("tests/ttnn/unit_tests/operations/fused/input_tensor.txt")
arg0 = ttnn.from_torch(torch_arg1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

# 3. Create Constants (%main_const_eval_0 and %main_const_eval_1)
# These represent the dummy mean (0.0) and var (1.0) for the standardization
stats_shape = [152]
const_0 = ttnn.full(shape=stats_shape, fill_value=0.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
const_1 = ttnn.full(shape=stats_shape, fill_value=1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# 4. Transform Input Layout and Reshape (%2 and %3)
# Move arg0 to TILE layout and reshape to [1, 152, 24, 32]
val_2 = ttnn.to_layout(arg0, layout=ttnn.TILE_LAYOUT)
# Note: ttnn.deallocate(arg0) happens automatically in Python when refs drop, or use ttnn.deallocate(arg0)
val_3 = ttnn.reshape(val_2, [1, 152, 24, 32])

# 5. Reshape Constants (%4 and %5)
# Reshape from [152] to [1, 152, 1, 1]
val_4 = ttnn.reshape(const_0, [1, 152, 1, 1])
val_5 = ttnn.reshape(const_1, [1, 152, 1, 1])

# Seperate tensors for Running Mean and Running Var
val_6 = ttnn.reshape(const_0, [1, 152, 1, 1])
val_7 = ttnn.reshape(const_1, [1, 152, 1, 1])

# 6. Execute Batch Norm Training (%6)
# Config matches: math_fidelity=hifi4, fp32_dest_acc_en=True, eps=9.99999996e-13, momentum=1.0
compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
)

output = ttnn.batch_norm(
    input=val_3,
    weight=val_4,  # Used as 'gamma' in the op, but graph passes 0s
    bias=val_5,  # Used as 'beta' in the op, but graph passes 1s
    running_mean=val_7,  # Graph passes %5 (ones)
    running_var=val_6,  # Graph passes %4 (zeros)
    eps=9.99999996e-13,
    momentum=1.0,
    compute_kernel_config=compute_config,
    training=True,
)

# Output result
output = ttnn.to_torch(output).to("cpu")
print("========== TTNN OUTPUT ==========")
print(f"max: {output.max().item()}")
print(f"min: {output.min().item()}")
print(f"std: {output.std().item()}")
print(f"mean: {output.mean().item()}")

torch_input = torch_arg1.reshape(1, 152, 24, 32).to(torch.bfloat16)

torch_val_4 = torch.zeros(stats_shape, dtype=torch.bfloat16)
torch_val_5 = torch.ones(stats_shape, dtype=torch.bfloat16)

# Seperate tensors for Running Mean and Running Var
torch_val_6 = torch.zeros(stats_shape, dtype=torch.bfloat16)
torch_val_7 = torch.ones(stats_shape, dtype=torch.bfloat16)

ref = torch.nn.functional.batch_norm(
    input=torch_input,
    running_mean=torch_val_7,
    running_var=torch_val_6,
    weight=torch_val_4,
    bias=torch_val_5,
    training=True,
    momentum=1.0,
    eps=9.99999996e-13,
)

print("========== TORCH OUTPUT ==========")
print(f"max: {ref.max().item()}")
print(f"min: {ref.min().item()}")
print(f"std: {ref.std().item()}")
print(f"mean: {ref.mean().item()}")

_, pcc = comp_pcc(ref, output)
max_atol = torch.max(torch.abs(ref - output)).item()
max_rtol = torch.max(torch.abs(ref - output) / torch.abs(output)).item()
rel_frobenius, _ = comp_relative_frobenius(ref, output)
print("========== COMPARISON ==========")
print(f"PCC: {pcc}")
print(f"Max ATOL: {max_atol}")
print(f"Max RTOL: {max_rtol}")
print(f"Relative Frobenius: {rel_frobenius}")

ttnn.close_device(device)
