"""Standalone script to export ActorCriticSequence checkpoint to ONNX format.

This script does NOT require Isaac Sim — it only needs PyTorch and ONNX.

Usage:
    conda activate rl_wheel_legged
    cd /home/sf4/Workspace/rm/rl_wheel_legged/IRobot_wl/IRobot_wl

    # Export latest model from a run directory:
    python scripts/rsl_rl/export_onnx.py \
        --run_dir logs/rsl_rl/wl_vmc_flat/2026-06-13_17-50-52 \
        --checkpoint model_400.pt

    # Specify output path:
    python scripts/rsl_rl/export_onnx.py \
        --run_dir logs/rsl_rl/wl_vmc_flat/2026-06-13_17-50-52 \
        --checkpoint model_400.pt \
        --output exported/policy.onnx

    # Export with latent output (for critic/visualization):
    python scripts/rsl_rl/export_onnx.py \
        --run_dir logs/rsl_rl/wl_vmc_flat/2026-06-13_17-50-52 \
        --checkpoint model_400.pt \
        --export_latent
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

# Add parent scripts dir to path for wl_sequence import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from wl_sequence.actor_critic_sequence import ActorCriticSequence, get_activation


class InferenceWrapper(nn.Module):
    """Wraps ActorCriticSequence.act_inference for ONNX export.

    The wrapper provides a clean forward(observations, observation_history) ->
    actions_mean interface with proper ONNX input/output names.
    """

    def __init__(self, actor_critic: ActorCriticSequence, export_latent: bool = False):
        super().__init__()
        self.encoder = actor_critic.encoder
        self.actor = actor_critic.actor
        self.export_latent = export_latent

    def forward(self, observations: torch.Tensor, observation_history: torch.Tensor):
        latent = self.encoder(observation_history)
        actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
        if self.export_latent:
            return actions_mean, latent
        return actions_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ActorCriticSequence checkpoint to ONNX"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., logs/rsl_rl/wl_vmc_flat/2026-06-13_17-50-52)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_400.pt",
        help="Checkpoint file name (default: model_400.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: <run_dir>/exported/policy.onnx)",
    )
    parser.add_argument(
        "--export_latent",
        action="store_true",
        help="Also export latent vector as second output",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed ONNX export logs",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> ActorCriticSequence:
    """Load state dict and reconstruct the model architecture.

    Infers model dimensions directly from the checkpoint weights.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    # --- Infer architecture from state_dict shapes ---
    # encoder: 135 -> 128 -> 64 -> latent_dim
    encoder_l0_weight = sd["encoder.0.weight"]  # (128, num_encoder_obs)
    encoder_l2_weight = sd["encoder.2.weight"]  # (64, 128)
    encoder_l4_weight = sd["encoder.4.weight"]  # (latent_dim, 64)

    num_encoder_obs = encoder_l0_weight.shape[1]
    encoder_hidden_dims = [encoder_l0_weight.shape[0], encoder_l2_weight.shape[0]]
    latent_dim = encoder_l4_weight.shape[0]

    # actor: (num_obs + latent) -> ... -> num_actions
    actor_l0_weight = sd["actor.0.weight"]  # (128, num_obs + latent)
    actor_l6_weight = sd["actor.6.weight"]  # (num_actions, 32)
    actor_input_dim = actor_l0_weight.shape[1]
    num_obs = actor_input_dim - latent_dim
    num_actions = actor_l6_weight.shape[0]

    # Infer actor hidden dims from weights
    actor_hidden_dims = []
    i = 0
    while f"actor.{i}.weight" in sd:
        actor_hidden_dims.append(sd[f"actor.{i}.weight"].shape[0])
        i += 2
    actor_hidden_dims = actor_hidden_dims[:-1]  # drop output layer dim

    # critic: num_critic_obs -> ... -> 1
    # Note: num_critic_obs passed to the constructor ALREADY includes latent_dim
    # (see WlSequenceRunner: obs.get(critic_group).shape[1] + latent_dim)
    critic_l0_weight = sd["critic.0.weight"]
    num_critic_obs = critic_l0_weight.shape[1]

    critic_hidden_dims = []
    i = 0
    while f"critic.{i}.weight" in sd:
        critic_hidden_dims.append(sd[f"critic.{i}.weight"].shape[0])
        i += 2
    critic_hidden_dims = critic_hidden_dims[:-1]  # drop output layer dim

    # --- Detect activation from agent.yaml if available ---
    # Default to elu (standard for this project)
    activation = "elu"
    params_dir = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.isfile(params_dir):
        try:
            import yaml
            with open(params_dir) as f:
                cfg = yaml.safe_load(f)
            activation = cfg.get("policy", {}).get("activation", "elu")
        except Exception:
            pass

    init_noise_std = float(sd["std"].mean().item())

    print(f"[INFO] Inferred architecture from checkpoint:")
    print(f"       num_obs = {num_obs}")
    print(f"       num_critic_obs = {num_critic_obs}")
    print(f"       num_encoder_obs = {num_encoder_obs}")
    print(f"       num_actions = {num_actions}")
    print(f"       latent_dim = {latent_dim}")
    print(f"       encoder_hidden_dims = {encoder_hidden_dims}")
    print(f"       actor_hidden_dims = {actor_hidden_dims}")
    print(f"       critic_hidden_dims = {critic_hidden_dims}")
    print(f"       activation = {activation}")
    print(f"       init_noise_std = {init_noise_std:.4f}")

    model = ActorCriticSequence(
        num_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        num_encoder_obs=num_encoder_obs,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
    ).to(device)

    model.load_state_dict(sd)
    model.eval()
    print(f"[INFO] Model loaded successfully from: {checkpoint_path}")
    return model


def export_to_onnx(
    model: ActorCriticSequence,
    output_path: str,
    export_latent: bool = False,
    opset_version: int = 18,
    verbose: bool = False,
):
    """Export the model to ONNX format."""
    wrapper = InferenceWrapper(model, export_latent=export_latent)
    wrapper.eval()
    wrapper.to("cpu")

    # Infer input shapes from model weights
    encoder_w0 = model.encoder[0].weight  # (128, num_encoder_obs)
    actor_w0 = model.actor[0].weight       # (128, num_obs + latent)
    num_encoder_obs = encoder_w0.shape[1]
    actor_input_dim = actor_w0.shape[1]
    num_obs = actor_input_dim - model.latent_dim

    batch_size = 1
    dummy_obs = torch.zeros(batch_size, num_obs, dtype=torch.float32)
    dummy_history = torch.zeros(batch_size, num_encoder_obs, dtype=torch.float32)

    # Dynamic axes for batch dimension
    dynamic_axes = {
        "observations": {0: "batch_size"},
        "observation_history": {0: "batch_size"},
        "actions": {0: "batch_size"},
    }
    if export_latent:
        dynamic_axes["latent"] = {0: "batch_size"}

    output_names = ["actions", "latent"] if export_latent else ["actions"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"[INFO] Exporting ONNX with:")
    print(f"       observations shape:  (batch, {num_obs})")
    print(f"       observation_history shape: (batch, {num_encoder_obs})")
    print(f"       output: {output_names}")
    print(f"       opset: {opset_version}")

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_history),
        output_path,
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        input_names=["observations", "observation_history"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"[INFO] ONNX model exported to: {output_path}")

    # --- Verification ---
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[INFO] ONNX model verification passed.")

    # Print I/O info
    print(f"[INFO] Model inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else "dynamic" for d in inp.type.tensor_type.shape.dim]
        print(f"       {inp.name}: {shape} ({inp.type.tensor_type.elem_type})")
    print(f"[INFO] Model outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else "dynamic" for d in out.type.tensor_type.shape.dim]
        print(f"       {out.name}: {shape} ({out.type.tensor_type.elem_type})")

    # --- Sanity check: compare PyTorch vs ONNX outputs when ONNXRuntime is available. ---
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("[WARN] onnxruntime is not installed here; skipped PyTorch-vs-ONNX numeric sanity check.")
    else:
        with torch.inference_mode():
            pt_result = wrapper(dummy_obs, dummy_history)

        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {
            "observations": dummy_obs.numpy(),
            "observation_history": dummy_history.numpy(),
        }
        if export_latent:
            ort_actions, ort_latent = ort_session.run(None, ort_inputs)
            max_diff_act = abs(pt_result[0].numpy() - ort_actions).max()
            max_diff_lat = abs(pt_result[1].numpy() - ort_latent).max()
            print(f"[INFO] Sanity check (dummy input):")
            print(f"       max actions diff: {max_diff_act:.2e}")
            print(f"       max latent diff:  {max_diff_lat:.2e}")
        else:
            (ort_actions,) = ort_session.run(None, ort_inputs)
            max_diff_act = abs(pt_result.numpy() - ort_actions).max()
            print(f"[INFO] Sanity check (dummy input):")
            print(f"       max actions diff: {max_diff_act:.2e}")

    return output_path


def main():
    args = parse_args()

    checkpoint_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if args.output is None:
        output_path = os.path.join(args.run_dir, "exported", "policy.onnx")
    else:
        output_path = args.output

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output:    {output_path}")

    model = load_model_from_checkpoint(checkpoint_path, device="cpu")
    export_to_onnx(
        model,
        output_path,
        export_latent=args.export_latent,
        opset_version=args.opset,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
