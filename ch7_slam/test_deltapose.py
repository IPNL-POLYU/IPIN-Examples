#!/usr/bin/env python3
import argparse
import sys
import numpy as np

# 你需要按你的仓库结构改这两行 import
from core.slam import (
    se2_apply,
    se2_compose,
    se2_relative,
    icp_point_to_point,
    create_pose_graph,
    se2_inverse,
    wrap_angle,
)


def se2_relative(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """如果你项目里已有 se2_relative 可直接 import，删掉这个函数。"""
    return se2_compose(se2_inverse(p_from), p_to)


def pose_error(gt: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    """返回平移误差 (m) 和角度误差 (rad)"""
    trans_err = float(np.linalg.norm(est[:2] - gt[:2]))
    yaw_err = float(abs(wrap_angle(est[2] - gt[2])))
    return trans_err, yaw_err


def make_scan(
    n: int,
    shape: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if shape == "arc":
        # 一段圆弧，比较像 2D LiDAR 扫描
        angles = rng.uniform(-1.8, 1.8, size=n)
        radius = rng.uniform(4.0, 8.0, size=n)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts = np.stack([x, y], axis=1)

        # 加一点结构，让点云不是完全对称，避免旋转退化
        wall_x = rng.uniform(-2.0, 2.0, size=n // 5)
        wall_y = rng.uniform(6.0, 7.0, size=n // 5)
        wall = np.stack([wall_x, wall_y], axis=1)
        pts = np.concatenate([pts, wall], axis=0)
        return pts.astype(np.float64)

    if shape == "box":
        # 矩形边界点
        t = rng.uniform(0.0, 1.0, size=n)
        side = rng.integers(0, 4, size=n)
        pts = np.zeros((n, 2), dtype=np.float64)
        w, h = 10.0, 6.0
        for i in range(n):
            if side[i] == 0:
                pts[i] = [w * (t[i] - 0.5), -h / 2]
            elif side[i] == 1:
                pts[i] = [w * (t[i] - 0.5), h / 2]
            elif side[i] == 2:
                pts[i] = [-w / 2, h * (t[i] - 0.5)]
            else:
                pts[i] = [w / 2, h * (t[i] - 0.5)]
        return pts

    raise ValueError(f"Unknown shape: {shape}")


def add_noise_and_outliers(
    pts: np.ndarray,
    noise_std: float,
    outlier_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    noisy = pts + rng.normal(0.0, noise_std, size=pts.shape)

    if outlier_ratio <= 0.0:
        return noisy

    k = int(len(noisy) * outlier_ratio)
    if k <= 0:
        return noisy

    outliers = rng.uniform([-15.0, -15.0], [15.0, 15.0], size=(k, 2))
    return np.concatenate([noisy, outliers], axis=0)


def run_case(
    name: str,
    source: np.ndarray,
    gt_pose: np.ndarray,
    init_pose: np.ndarray,
    max_corr_dist: float | None,
    tol_xy: float,
    tol_yaw_deg: float,
    max_iter: int,
) -> bool:
    target = se2_apply(gt_pose, source)

    # 模拟噪声与外点
    source_obs = add_noise_and_outliers(source, noise_std=0.01, outlier_ratio=0.05, seed=1)
    target_obs = add_noise_and_outliers(target, noise_std=0.01, outlier_ratio=0.05, seed=2)

    est, iters, residual, converged = icp_point_to_point(
        source_obs,
        target_obs,
        initial_pose=init_pose,
        max_iterations=max_iter,
        tolerance=1e-4,
        max_correspondence_distance=max_corr_dist,
        min_correspondences=20,
    )

    trans_err, yaw_err = pose_error(gt_pose, est)
    yaw_err_deg = yaw_err * 180.0 / np.pi

    print(f"\n[{name}]")
    print(f"  gt_pose   : {gt_pose}")
    print(f"  init_pose : {init_pose}")
    print(f"  est_pose  : {est}")
    print(f"  converged : {converged}, iters={iters}, residual={residual:.6f}")
    print(f"  error     : trans={trans_err:.4f} m, yaw={yaw_err_deg:.3f} deg")

    ok = (trans_err <= tol_xy) and (yaw_err_deg <= tol_yaw_deg) and converged
    print(f"  PASS      : {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--max_corr", type=float, default=0.7, help="max correspondence distance, meters")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    source = make_scan(n=400, shape="arc", seed=args.seed)

    # Case 1: 小位移小旋转，比较容易
    gt1 = np.array([0.8, -0.4, np.deg2rad(5.0)], dtype=np.float64)
    init1 = gt1 + np.array([0.2, -0.2, np.deg2rad(3.0)], dtype=np.float64)

    # Case 2: 明显旋转 + 平移，用来抓更新方向问题
    gt2 = np.array([2.0, 1.0, np.deg2rad(35.0)], dtype=np.float64)
    init2 = gt2 + np.array([-0.5, 0.3, np.deg2rad(-10.0)], dtype=np.float64)

    # Case 3: 用 poses 形式构造 se2_relative 的 initial_guess
    pose_i = np.array([1.0, 2.0, np.deg2rad(20.0)], dtype=np.float64)
    pose_j = se2_compose(pose_i, np.array([1.2, -0.7, np.deg2rad(30.0)], dtype=np.float64))
    init3 = se2_relative(pose_i, pose_j)
    gt3 = init3.copy()  # 这里 gt 就设成相对位姿

    ok1 = run_case(
        "case1_small_motion",
        source=source,
        gt_pose=gt1,
        init_pose=init1,
        max_corr_dist=args.max_corr,
        tol_xy=0.10,
        tol_yaw_deg=2.0,
        max_iter=args.max_iter,
    )

    ok2 = run_case(
        "case2_rotation_motion",
        source=source,
        gt_pose=gt2,
        init_pose=init2,
        max_corr_dist=args.max_corr,
        tol_xy=0.15,
        tol_yaw_deg=3.0,
        max_iter=args.max_iter,
    )

    # 这个 case 的目标就是验证 se2_relative 形式的初值能正常工作
    ok3 = run_case(
        "case3_relative_initial_guess",
        source=source,
        gt_pose=gt3,
        init_pose=init3 + np.array([0.1, -0.1, np.deg2rad(2.0)], dtype=np.float64),
        max_corr_dist=args.max_corr,
        tol_xy=0.10,
        tol_yaw_deg=2.0,
        max_iter=args.max_iter,
    )

    all_ok = ok1 and ok2 and ok3
    print("\nALL PASS:", all_ok)

    # assert 风格，CI 里好用
    if not all_ok:
        print("\nSome cases failed. If case2 fails badly, check ICP pose update order (left-multiply vs right-multiply).")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
