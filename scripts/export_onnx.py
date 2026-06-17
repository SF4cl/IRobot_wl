#!/usr/bin/env python3
"""Export the WlSequenceRunner ActorCriticSequence model to ONNX format.

This script loads a checkpoint and exports the inference portion of the network
(encoder + actor) to ONNX. The exported model takes two inputs:
  - observations: (1, num_obs)  -- current observation
  - observation_history: (1, num_encoder_obs) -- last 5 observations flattened
And produces one output:
  - actions: (1, num_actions) -- task-space action mean

All dimensions are auto-detected from the checkpoint state dict.

Usage:
    cd /home/sf4/Workspace/rm/rl_wheel_legged/IRobot_wl/IRobot_wl
    python scripts/export_onnx.py \\
        --checkpoint logs/rsl_rl/wl_vmc_flat/2026-06-17_16-29-55/model_700.pt
"""

import argparse
import os
import re
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

# Add the rsl_rl/wl_sequence module to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "rsl_rl", "wl_sequence"))
from actor_critic_sequence import ActorCriticSequence


def _mlp_layer_dims(state_dict: dict, prefix: str) -> list[int]:
    """Extract hidden dims and input/output dims from a sequential MLP in the state dict.

    The Sequential is built as: Linear, ELU, Linear, ELU, ..., Linear
    So weight keys are at even indices: prefix.0.weight, prefix.2.weight, ...

    Returns [input_dim, hidden_0, hidden_1, ..., output_dim].
    """
    weight_keys = sorted(
        [k for k in state_dict if k.startswith(f"{prefix}.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    dims = []
    for key in weight_keys:
        w = state_dict[key]
        dims.append(w.shape[1])  # input dim of this layer
    # Last layer's output dim
    dims.append(state_dict[weight_keys[-1]].shape[0])
    return dims


def infer_dims_from_state_dict(state_dict: dict) -> dict:
    """Infer all network dimensions from the checkpoint state dict."""
    encoder_dims = _mlp_layer_dims(state_dict, "encoder")
    actor_dims = _mlp_layer_dims(state_dict, "actor")
    critic_dims = _mlp_layer_dims(state_dict, "critic")

    num_encoder_obs = encoder_dims[0]
    encoder_hidden_dims = encoder_dims[1:-1]
    latent_dim = encoder_dims[-1]

    actor_input_dim = actor_dims[0]
    num_obs = actor_input_dim - latent_dim
    actor_hidden_dims = actor_dims[1:-1]
    num_actions = actor_dims[-1]

    num_critic_obs = critic_dims[0]
    critic_hidden_dims = critic_dims[1:-1]

    # Detect activation from available activations in the model
    activation = "elu"  # default

    # Infer init_noise_std from std parameter
    init_noise_std = float(state_dict["std"].mean().item()) if "std" in state_dict else 0.5

    return {
        "num_obs": num_obs,
        "num_critic_obs": num_critic_obs,
        "num_actions": num_actions,
        "num_encoder_obs": num_encoder_obs,
        "latent_dim": latent_dim,
        "encoder_hidden_dims": tuple(encoder_hidden_dims),
        "actor_hidden_dims": tuple(actor_hidden_dims),
        "critic_hidden_dims": tuple(critic_hidden_dims),
        "activation": activation,
        "init_noise_std": init_noise_std,
    }


class InferenceWrapper(nn.Module):
    """Wraps the ActorCriticSequence to expose act_inference as a single forward pass.

    ONNX export requires a single forward(observations, observation_history) method.
    """

    def __init__(self, model: ActorCriticSequence):
        super().__init__()
        self.model = model

    def forward(self, observations: torch.Tensor, observation_history: torch.Tensor):
        actions, latent = self.model.act_inference(observations, observation_history)
        return actions


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    opset_version: int = 17,
) -> None:
    """Load a checkpoint and export the policy to ONNX."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    iteration = checkpoint.get("iter", "unknown")
    print(f"[INFO] Checkpoint from iteration: {iteration}")

    # Auto-detect dimensions from state dict
    dims = infer_dims_from_state_dict(state_dict)
    print("[INFO] Detected network dimensions:")
    for key, value in dims.items():
        print(f"  {key}: {value}")

    # Build model
    model = ActorCriticSequence(**dims).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Model weights loaded successfully.")

    # Wrap for ONNX export
    wrapper = InferenceWrapper(model)
    wrapper.eval()

    # Create dummy inputs using detected dimensions
    dummy_obs = torch.randn(1, dims["num_obs"], device=device)
    dummy_history = torch.randn(1, dims["num_encoder_obs"], device=device)

    # Test forward pass before export
    with torch.no_grad():
        test_output = wrapper(dummy_obs, dummy_history)
    print(f"[INFO] Test forward pass output shape: {test_output.shape}")
    print(f"[INFO] Test forward pass output: {test_output.squeeze().tolist()}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    print(f"[INFO] Exporting ONNX to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_history),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observations", "observation_history"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch_size"},
            "observation_history": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )
    print("[INFO] ONNX export completed successfully.")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model verification passed.")

    # Run a quick inference comparison
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    ort_inputs = {
        "observations": dummy_obs.cpu().numpy(),
        "observation_history": dummy_history.cpu().numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)
    print(f"[INFO] ONNX Runtime output shape: {ort_outputs[0].shape}")
    print(f"[INFO] ONNX Runtime output: {ort_outputs[0].squeeze().tolist()}")

    # Compare
    torch_output = test_output.cpu().numpy()
    onnx_output = ort_outputs[0]
    max_diff = abs(torch_output - onnx_output).max()
    print(f"[INFO] Max difference between PyTorch and ONNX outputs: {max_diff:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Export ActorCriticSequence model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/rsl_rl/wl_vmc_flat/2026-06-17_16-29-55/model_700.pt",
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the ONNX model. Defaults to <checkpoint_dir>/exported/<checkpoint_name>.onnx",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()

    if args.output is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(ckpt_dir, "exported", f"{ckpt_name}.onnx")

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
