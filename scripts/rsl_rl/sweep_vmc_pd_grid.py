"""Offline VMC geometry and PD grid check.

This script does not launch Isaac Sim.  It sweeps the available WL virtual-leg
workspace, checks FK/IK consistency, compares VMC Jacobian variants against
finite differences, and verifies whether a simple old-style task-space PD would
push L0/theta0 errors back toward the target.

Outputs are written as CSV/JSON plus a small self-contained HTML/SVG report.

Example:
    python scripts/rsl_rl/sweep_vmc_pd_grid.py
    python scripts/rsl_rl/sweep_vmc_pd_grid.py --grid-l0 121 --grid-theta0 121
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_L1 = 0.21665632675675972
DEFAULT_L2 = 0.2540023491164531
DEFAULT_OFFSET = -0.007712217793726145
DEFAULT_THETA1_OFFSET = 0.14299916248023697
DEFAULT_THETA2_OFFSET = 2.406020345452543
DEFAULT_L0_MIN = 0.1219258562330587
DEFAULT_L0_MAX = 0.3006386827708927


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def forward_kinematics(theta1: float, theta2: float, l1: float, l2: float, offset: float) -> tuple[float, float]:
    end_x = offset + l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    end_y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    l0 = math.hypot(end_x, end_y)
    theta0 = math.atan2(end_y, end_x) - math.pi / 2.0
    return l0, wrap_to_pi(theta0)


def inverse_kinematics(theta0: float, l0: float, l1: float, l2: float, offset: float) -> tuple[float, float]:
    gamma = theta0 + math.pi / 2.0
    target_x = l0 * math.cos(gamma) - offset
    target_y = l0 * math.sin(gamma)
    target_len = math.hypot(target_x, target_y)
    target_len = min(max(target_len, abs(l1 - l2) + 1.0e-4), l1 + l2 - 1.0e-4)

    cos_beta = (l1 * l1 + l2 * l2 - target_len * target_len) / (2.0 * l1 * l2)
    cos_beta = min(max(cos_beta, -1.0 + 1.0e-6), 1.0 - 1.0e-6)
    beta = math.acos(cos_beta)
    theta2 = math.pi - beta

    alpha = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = math.atan2(target_y, target_x) - alpha
    return theta1, theta2


def finite_difference_jacobian(
    theta1: float,
    theta2: float,
    l1: float,
    l2: float,
    offset: float,
    eps: float = 1.0e-6,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return task Jacobian J where rows are [L0, theta0] and columns [q1, q2]."""

    def grad(coord_index: int, joint_index: int) -> float:
        q_plus = [theta1, theta2]
        q_minus = [theta1, theta2]
        q_plus[joint_index] += eps
        q_minus[joint_index] -= eps
        plus = forward_kinematics(q_plus[0], q_plus[1], l1, l2, offset)[coord_index]
        minus = forward_kinematics(q_minus[0], q_minus[1], l1, l2, offset)[coord_index]
        if coord_index == 1:
            return wrap_to_pi(plus - minus) / (2.0 * eps)
        return (plus - minus) / (2.0 * eps)

    return (
        (grad(0, 0), grad(0, 1)),
        (grad(1, 0), grad(1, 1)),
    )


def current_jacobian_transpose_coeffs(
    theta1: float, theta2: float, l1: float, l2: float, offset: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Rows [hip, knee], columns [F_L0, Tp_theta0], matching current vmc.py."""
    l0, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)
    gamma = theta0 + math.pi / 2.0
    l0_safe = max(l0, 0.05)
    d_l_d_q1 = l1 * math.sin(gamma - theta1) - l2 * math.sin(theta1 + theta2 - gamma)
    d_l_d_q2 = -l2 * math.sin(theta1 + theta2 - gamma)
    d_theta_d_q1 = (
        l1 * math.cos(gamma - theta1) + l2 * math.cos(theta1 + theta2 - gamma)
    ) / l0_safe
    d_theta_d_q2 = l2 * math.cos(theta1 + theta2 - gamma) / l0_safe
    return ((d_l_d_q1, d_theta_d_q1), (d_l_d_q2, d_theta_d_q2))


def pre_fb627e5_coeffs(
    theta1: float, theta2: float, l1: float, l2: float, offset: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Rows [hip, knee], columns [F_L0, Tp], matching vmc.py before fb627e5."""
    l0, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)
    theta0_shifted = theta0 + math.pi / 2.0
    l0_safe = max(l0, 0.05)
    t11 = l1 * math.sin(theta0_shifted - theta1) - l2 * math.sin(theta1 + theta2 - theta0_shifted)
    t12 = (l1 * math.cos(theta0_shifted - theta1) - l2 * math.cos(theta1 + theta2 - theta0_shifted)) / l0_safe
    t21 = -l2 * math.sin(theta1 + theta2 - theta0_shifted)
    t22 = -l2 * math.cos(theta1 + theta2 - theta0_shifted) / l0_safe
    return ((t11, -t12), (t21, -t22))


def matmul_2x2_vec(m: tuple[tuple[float, float], tuple[float, float]], v: tuple[float, float]) -> tuple[float, float]:
    return (m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1])


def transpose_coeffs_to_jacobian(coeffs: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Convert rows [q1, q2], cols [L, theta] to J rows [L, theta], cols [q1, q2]."""
    return ((coeffs[0][0], coeffs[1][0]), (coeffs[0][1], coeffs[1][1]))


def max_abs_diff_jt(
    coeffs: tuple[tuple[float, float], tuple[float, float]],
    jacobian: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    return max(
        abs(coeffs[0][0] - jacobian[0][0]),
        abs(coeffs[1][0] - jacobian[0][1]),
        abs(coeffs[0][1] - jacobian[1][0]),
        abs(coeffs[1][1] - jacobian[1][1]),
    )


def task_velocity_proxy(
    coeffs: tuple[tuple[float, float], tuple[float, float]],
    generalized_force: tuple[float, float],
) -> tuple[float, float]:
    """Use unit joint inertia proxy: qdot = tau, xdot = J qdot."""
    tau = matmul_2x2_vec(coeffs, generalized_force)
    jacobian = transpose_coeffs_to_jacobian(coeffs)
    return matmul_2x2_vec(jacobian, tau)


def linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [(lo + hi) * 0.5]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def color_ramp(value: float, lo: float, hi: float, diverging: bool = False) -> str:
    if hi <= lo:
        t = 0.0
    else:
        t = clamp((value - lo) / (hi - lo), 0.0, 1.0)
    if diverging:
        # blue -> white -> red
        if t < 0.5:
            u = t * 2.0
            r = int(42 + (245 - 42) * u)
            g = int(96 + (245 - 96) * u)
            b = int(180 + (245 - 180) * u)
        else:
            u = (t - 0.5) * 2.0
            r = int(245 + (190 - 245) * u)
            g = int(245 + (48 - 245) * u)
            b = int(245 + (48 - 245) * u)
        return f"#{r:02x}{g:02x}{b:02x}"
    r = int(246 - 210 * t)
    g = int(248 - 142 * t)
    b = int(255 - 89 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(
    path: Path,
    title: str,
    xs: list[float],
    ys: list[float],
    values: list[list[float]],
    lo: float | None = None,
    hi: float | None = None,
    diverging: bool = False,
) -> None:
    finite_values = [v for row in values for v in row if math.isfinite(v)]
    if lo is None:
        lo = min(finite_values) if finite_values else 0.0
    if hi is None:
        hi = max(finite_values) if finite_values else 1.0
    cell = 5
    margin_l = 70
    margin_t = 38
    margin_b = 42
    width = margin_l + len(xs) * cell + 20
    height = margin_t + len(ys) * cell + margin_b
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:11px;fill:#222}</style>',
        f'<text x="{margin_l}" y="20" font-size="14" font-weight="700">{title}</text>',
        f'<text x="{margin_l}" y="{height - 12}">theta0 range: {xs[0]:.3f} to {xs[-1]:.3f} rad</text>',
        f'<text x="8" y="{margin_t + 12}">L0 {ys[-1]:.3f}</text>',
        f'<text x="8" y="{margin_t + len(ys) * cell - 2}">L0 {ys[0]:.3f}</text>',
    ]
    for yi, row in enumerate(values):
        draw_y = margin_t + (len(ys) - 1 - yi) * cell
        for xi, value in enumerate(row):
            fill = "#d9d9d9" if not math.isfinite(value) else color_ramp(value, lo, hi, diverging)
            x = margin_l + xi * cell
            parts.append(f'<rect x="{x}" y="{draw_y}" width="{cell}" height="{cell}" fill="{fill}"/>')
    parts.append(
        f'<text x="{width - 150}" y="20">min={lo:.3g}, max={hi:.3g}</text>'
    )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report_html(out_dir: Path, summary: dict, image_names: list[str]) -> None:
    cards = "\n".join(
        f'<section><h2>{name}</h2><img src="{name}" alt="{name}"></section>' for name in image_names
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WL VMC PD Grid Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #202124; }}
    code, pre {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
    section {{ margin: 24px 0; }}
    img {{ border: 1px solid #ddd; max-width: 100%; image-rendering: pixelated; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>WL VMC PD Grid Report</h1>
  <p>Generated by <code>scripts/rsl_rl/sweep_vmc_pd_grid.py</code>.</p>
  <h2>Summary</h2>
  <pre>{json.dumps(summary, indent=2)}</pre>
  {cards}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep WL VMC L0/theta0 workspace and check PD/Jacobian behavior.")
    parser.add_argument("--out", type=Path, default=Path("scripts/rsl_rl/vmc_pd_grid_report"))
    parser.add_argument("--grid-l0", type=int, default=91)
    parser.add_argument("--grid-theta0", type=int, default=121)
    parser.add_argument("--l1", type=float, default=DEFAULT_L1)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET)
    parser.add_argument("--l0-min", type=float, default=DEFAULT_L0_MIN)
    parser.add_argument("--l0-max", type=float, default=DEFAULT_L0_MAX)
    parser.add_argument("--theta0-min", type=float, default=-0.75)
    parser.add_argument("--theta0-max", type=float, default=0.75)
    parser.add_argument("--kp-theta", type=float, default=50.0)
    parser.add_argument("--kp-l0", type=float, default=900.0)
    parser.add_argument("--l0-eps", type=float, default=0.01)
    parser.add_argument("--theta0-eps", type=float, default=0.05)
    parser.add_argument("--feedforward-force", type=float, default=40.0)
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    l0_values = linspace(args.l0_min, args.l0_max, args.grid_l0)
    theta0_values = linspace(args.theta0_min, args.theta0_max, args.grid_theta0)
    rows: list[dict[str, float | int]] = []

    maps: dict[str, list[list[float]]] = {
        "reachable": [],
        "fk_l0_error": [],
        "fk_theta0_error": [],
        "current_jacobian_error": [],
        "pre_fb627e5_jacobian_error": [],
        "current_force_extension": [],
        "pre_force_extension": [],
        "old_pd_l0_restoring": [],
        "old_pd_theta_restoring": [],
        "feedforward_tau_norm": [],
    }

    for l0_target in l0_values:
        row_maps = {name: [] for name in maps}
        for theta0_target in theta0_values:
            theta1, theta2 = inverse_kinematics(theta0_target, l0_target, args.l1, args.l2, args.offset)
            fk_l0, fk_theta0 = forward_kinematics(theta1, theta2, args.l1, args.l2, args.offset)
            fk_l0_error = fk_l0 - l0_target
            fk_theta0_error = wrap_to_pi(fk_theta0 - theta0_target)
            reachable = 1.0 if abs(fk_l0_error) < 2.0e-3 and abs(fk_theta0_error) < 2.0e-3 else 0.0

            fd_j = finite_difference_jacobian(theta1, theta2, args.l1, args.l2, args.offset)
            cur = current_jacobian_transpose_coeffs(theta1, theta2, args.l1, args.l2, args.offset)
            pre = pre_fb627e5_coeffs(theta1, theta2, args.l1, args.l2, args.offset)
            cur_j_err = max_abs_diff_jt(cur, fd_j)
            pre_j_err = max_abs_diff_jt(pre, fd_j)

            cur_force_extension = task_velocity_proxy(cur, (1.0, 0.0))[0]
            pre_force_extension = task_velocity_proxy(pre, (1.0, 0.0))[0]

            # Old action semantics: target L0/theta0 + task-space PD.
            below_l0 = max(args.l0_min, l0_target - args.l0_eps)
            above_l0 = min(args.l0_max, l0_target + args.l0_eps)
            theta1_b, theta2_b = inverse_kinematics(theta0_target, below_l0, args.l1, args.l2, args.offset)
            theta1_a, theta2_a = inverse_kinematics(theta0_target, above_l0, args.l1, args.l2, args.offset)
            cur_b = current_jacobian_transpose_coeffs(theta1_b, theta2_b, args.l1, args.l2, args.offset)
            cur_a = current_jacobian_transpose_coeffs(theta1_a, theta2_a, args.l1, args.l2, args.offset)
            l0_b, _ = forward_kinematics(theta1_b, theta2_b, args.l1, args.l2, args.offset)
            l0_a, _ = forward_kinematics(theta1_a, theta2_a, args.l1, args.l2, args.offset)
            f_b = args.kp_l0 * (l0_target - l0_b)
            f_a = args.kp_l0 * (l0_target - l0_a)
            l0_dot_b = task_velocity_proxy(cur_b, (f_b, 0.0))[0]
            l0_dot_a = task_velocity_proxy(cur_a, (f_a, 0.0))[0]
            old_pd_l0_restoring = min(l0_dot_b, -l0_dot_a)

            below_theta = theta0_target - args.theta0_eps
            above_theta = theta0_target + args.theta0_eps
            theta1_tb, theta2_tb = inverse_kinematics(below_theta, l0_target, args.l1, args.l2, args.offset)
            theta1_ta, theta2_ta = inverse_kinematics(above_theta, l0_target, args.l1, args.l2, args.offset)
            cur_tb = current_jacobian_transpose_coeffs(theta1_tb, theta2_tb, args.l1, args.l2, args.offset)
            cur_ta = current_jacobian_transpose_coeffs(theta1_ta, theta2_ta, args.l1, args.l2, args.offset)
            _, theta_b = forward_kinematics(theta1_tb, theta2_tb, args.l1, args.l2, args.offset)
            _, theta_a = forward_kinematics(theta1_ta, theta2_ta, args.l1, args.l2, args.offset)
            tp_b = args.kp_theta * wrap_to_pi(theta0_target - theta_b)
            tp_a = args.kp_theta * wrap_to_pi(theta0_target - theta_a)
            theta_dot_b = task_velocity_proxy(cur_tb, (0.0, tp_b))[1]
            theta_dot_a = task_velocity_proxy(cur_ta, (0.0, tp_a))[1]
            old_pd_theta_restoring = min(theta_dot_b, -theta_dot_a)

            ff_tau = matmul_2x2_vec(cur, (args.feedforward_force, 0.0))
            feedforward_tau_norm = math.hypot(ff_tau[0], ff_tau[1])

            values = {
                "reachable": reachable,
                "fk_l0_error": abs(fk_l0_error),
                "fk_theta0_error": abs(fk_theta0_error),
                "current_jacobian_error": cur_j_err,
                "pre_fb627e5_jacobian_error": pre_j_err,
                "current_force_extension": cur_force_extension,
                "pre_force_extension": pre_force_extension,
                "old_pd_l0_restoring": old_pd_l0_restoring,
                "old_pd_theta_restoring": old_pd_theta_restoring,
                "feedforward_tau_norm": feedforward_tau_norm,
            }
            for name, value in values.items():
                row_maps[name].append(value)
            rows.append(
                {
                    "l0_target": l0_target,
                    "theta0_target": theta0_target,
                    "theta1": theta1,
                    "theta2": theta2,
                    "fk_l0": fk_l0,
                    "fk_theta0": fk_theta0,
                    **values,
                }
            )
        for name in maps:
            maps[name].append(row_maps[name])

    csv_path = out_dir / "grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(name: str) -> dict[str, float]:
        vals = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}

    summary = {
        "grid": {"l0": args.grid_l0, "theta0": args.grid_theta0},
        "ranges": {
            "l0": [args.l0_min, args.l0_max],
            "theta0": [args.theta0_min, args.theta0_max],
        },
        "stats": {name: stats(name) for name in maps},
        "notes": [
            "current_jacobian_error compares current vmc.py coefficients to finite-difference FK Jacobian.",
            "pre_fb627e5_jacobian_error compares the mapping before commit fb627e5 to the same finite-difference Jacobian.",
            "old_pd_*_restoring should be positive if old target L0/theta0 PD locally pushes errors back toward target.",
            "force_extension should be positive if a positive axial F tends to increase L0 under a unit-inertia proxy.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    image_specs = [
        ("reachable.svg", "IK reachable mask", "reachable", 0.0, 1.0, False),
        ("fk_l0_error.svg", "IK -> FK absolute L0 error [m]", "fk_l0_error", 0.0, None, False),
        ("fk_theta0_error.svg", "IK -> FK absolute theta0 error [rad]", "fk_theta0_error", 0.0, None, False),
        ("current_jacobian_error.svg", "Current Jacobian max error", "current_jacobian_error", 0.0, None, False),
        ("pre_fb627e5_jacobian_error.svg", "Pre-fb627e5 Jacobian max error", "pre_fb627e5_jacobian_error", 0.0, None, False),
        ("current_force_extension.svg", "Current positive F extension proxy", "current_force_extension", 0.0, None, False),
        ("pre_force_extension.svg", "Pre-fb627e5 positive F extension proxy", "pre_force_extension", None, None, True),
        ("old_pd_l0_restoring.svg", "Old target-PD L0 restoring proxy", "old_pd_l0_restoring", 0.0, None, False),
        ("old_pd_theta_restoring.svg", "Old target-PD theta0 restoring proxy", "old_pd_theta_restoring", 0.0, None, False),
        ("feedforward_tau_norm.svg", "Current 40N feedforward joint torque norm [Nm]", "feedforward_tau_norm", 0.0, None, False),
    ]
    image_names = []
    for filename, title, key, lo, hi, diverging in image_specs:
        write_heatmap_svg(out_dir / filename, title, theta0_values, l0_values, maps[key], lo, hi, diverging)
        image_names.append(filename)
    write_report_html(out_dir, summary, image_names)

    print(f"Wrote VMC PD grid report: {out_dir / 'index.html'}")
    print(json.dumps(summary["stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
