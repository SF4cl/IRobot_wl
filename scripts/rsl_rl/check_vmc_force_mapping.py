"""Check whether VMC F/Tp maps to joint torques consistently.

This script is intentionally Isaac-free and torch-free.  It compares the
current closed-form ``vmc_torques()`` mapping against a finite-difference
Jacobian of the same ``L0/theta0`` forward kinematics.

Run from the IRobot_wl project root:

    python scripts/rsl_rl/check_vmc_force_mapping.py

or inside the training environment:

    conda run -n isaacsim510 python scripts/rsl_rl/check_vmc_force_mapping.py
"""

from __future__ import annotations

import argparse
import math
import random


DEFAULT_L1 = 0.21665632675675972
DEFAULT_L2 = 0.2540023491164531
DEFAULT_OFFSET = -0.007712217793726145
DEFAULT_THETA1 = 0.14299916248023697
DEFAULT_THETA2 = 2.406020345452543


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def forward_kinematics(theta1: float, theta2: float, l1: float, l2: float, offset: float) -> tuple[float, float]:
    """Same single-leg FK used by IRobot_wl.tasks...mdp.vmc.forward_kinematics."""
    end_x = offset + l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    end_y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    l0 = math.sqrt(end_x * end_x + end_y * end_y)
    theta0 = math.atan2(end_y, end_x) - math.pi / 2
    return l0, theta0


def current_mapping_coefficients(
    theta1: float, theta2: float, l1: float, l2: float, offset: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return coefficients mapping [F, Tp] to [hip_tau, knee_tau] in current code."""
    l0, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)
    gamma = theta0 + math.pi / 2
    l0_safe = max(l0, 0.05)

    d_l_d_theta1 = l1 * math.sin(gamma - theta1) - l2 * math.sin(theta1 + theta2 - gamma)
    d_l_d_theta2 = -l2 * math.sin(theta1 + theta2 - gamma)
    d_theta_d_theta1 = (
        l1 * math.cos(gamma - theta1) + l2 * math.cos(theta1 + theta2 - gamma)
    ) / l0_safe
    d_theta_d_theta2 = l2 * math.cos(theta1 + theta2 - gamma) / l0_safe

    # Current vmc.py uses:
    #   T1 = dL/dtheta1 * F + dtheta/dtheta1 * Tp
    #   T2 = dL/dtheta2 * F + dtheta/dtheta2 * Tp
    return (d_l_d_theta1, d_theta_d_theta1), (d_l_d_theta2, d_theta_d_theta2)


def finite_difference_expected_coefficients(
    theta1: float,
    theta2: float,
    l1: float,
    l2: float,
    offset: float,
    eps: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return J_task^T coefficients for task coordinates [L0, theta0]."""

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

    d_l_d_q1 = grad(0, 0)
    d_l_d_q2 = grad(0, 1)
    d_theta_d_q1 = grad(1, 0)
    d_theta_d_q2 = grad(1, 1)

    # Expected virtual work relation:
    #   tau = dL/dq * F + dtheta/dq * Tp
    return (d_l_d_q1, d_theta_d_q1), (d_l_d_q2, d_theta_d_q2)


def mat_diff_abs_max(a: tuple[tuple[float, float], tuple[float, float]], b: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return max(abs(a[row][col] - b[row][col]) for row in range(2) for col in range(2))


def fmt_matrix(m: tuple[tuple[float, float], tuple[float, float]]) -> str:
    return (
        f"[[{m[0][0]: .9f}, {m[0][1]: .9f}],\n"
        f" [{m[1][0]: .9f}, {m[1][1]: .9f}]]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check WL VMC F/Tp to joint-torque mapping.")
    parser.add_argument("--l1", type=float, default=DEFAULT_L1)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET)
    parser.add_argument("--theta1", type=float, default=DEFAULT_THETA1, help="Single-leg mirrored theta1 to inspect.")
    parser.add_argument("--theta2", type=float, default=DEFAULT_THETA2, help="Single-leg mirrored theta2 to inspect.")
    parser.add_argument("--eps", type=float, default=1.0e-6, help="Finite-difference step.")
    parser.add_argument("--samples", type=int, default=1000, help="Random configurations to scan.")
    parser.add_argument("--tol", type=float, default=1.0e-3, help="Max allowed coefficient error for PASS.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)

    current = current_mapping_coefficients(args.theta1, args.theta2, args.l1, args.l2, args.offset)
    expected = finite_difference_expected_coefficients(args.theta1, args.theta2, args.l1, args.l2, args.offset, args.eps)
    diff = mat_diff_abs_max(current, expected)
    l0, theta0 = forward_kinematics(args.theta1, args.theta2, args.l1, args.l2, args.offset)

    print("=" * 78)
    print("VMC force mapping check at nominal mirrored joint state")
    print(f"theta1={args.theta1:.9f}, theta2={args.theta2:.9f}, L0={l0:.9f}, theta0={theta0:.9f}")
    print("Rows are [hip, knee], columns are [F, Tp].")
    print("-" * 78)
    print("Current vmc.py coefficients:")
    print(fmt_matrix(current))
    print("Finite-difference expected J_task^T coefficients:")
    print(fmt_matrix(expected))
    print(f"max abs coefficient error: {diff:.9f}")
    print("-" * 78)
    print("Pure F=1, Tp=0 torque [hip, knee]:")
    print(f"  current  [{current[0][0]: .9f}, {current[1][0]: .9f}]")
    print(f"  expected [{expected[0][0]: .9f}, {expected[1][0]: .9f}]")
    print("Pure F=0, Tp=1 torque [hip, knee]:")
    print(f"  current  [{current[0][1]: .9f}, {current[1][1]: .9f}]")
    print(f"  expected [{expected[0][1]: .9f}, {expected[1][1]: .9f}]")

    max_scan_diff = diff
    worst_state = (args.theta1, args.theta2)
    for _ in range(args.samples):
        # Scan a conservative bent-leg range around the WL mirrored convention.
        theta1 = random.uniform(-0.8, 0.8)
        theta2 = random.uniform(1.0, 2.9)
        cur = current_mapping_coefficients(theta1, theta2, args.l1, args.l2, args.offset)
        exp = finite_difference_expected_coefficients(theta1, theta2, args.l1, args.l2, args.offset, args.eps)
        scan_diff = mat_diff_abs_max(cur, exp)
        if scan_diff > max_scan_diff:
            max_scan_diff = scan_diff
            worst_state = (theta1, theta2)

    print("-" * 78)
    print(f"random scan samples: {args.samples}")
    print(f"max scan coefficient error: {max_scan_diff:.9f} at theta1={worst_state[0]:.9f}, theta2={worst_state[1]:.9f}")
    if max_scan_diff <= args.tol:
        print(f"PASS: current mapping matches the [L0, theta0] Jacobian within tol={args.tol:g}.")
        return 0
    print(f"FAIL: current mapping does not match the [L0, theta0] Jacobian within tol={args.tol:g}.")
    print("This is expected if Tp is not meant to be a true theta0 generalized torque.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
