import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from utils.config import Config

def clip_radially(pcd: o3d.geometry.PointCloud, radius: float = 2.0, step: float = 25.0) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    depths = np.linalg.norm(points, axis=1)
    
    min_depth = np.min(depths)
    max_depth = np.max(depths)

    keep_mask = np.zeros(len(points), dtype=bool)

    for d_start in np.arange(min_depth, max_depth, step):
        d_end = d_start + step

        bin_mask = (depths >= d_start) & (depths < d_end)
        bin_points = points[bin_mask]

        if bin_points.shape[0] < 3:
            continue

        xy = bin_points[:, :2]

        X = np.hstack((2*xy, np.ones((xy.shape[0], 1))))
        y = np.sum(xy**2, axis=1)

        try:
            model = RANSACRegressor()
            model.fit(X, y)
            D, E, F = model.estimator_.coef_[0], model.estimator_.coef_[1], model.estimator_.intercept_
            cx, cy = D, E
            r = np.sqrt(F + cx**2 + cy**2)

            distances = np.linalg.norm(xy - np.array([cx, cy]), axis=1)

            within_radius = np.abs(distances - r) < radius
            keep_mask[bin_mask] = within_radius
        except Exception as e:
            continue

    clipped_pcd = pcd.select_by_index(np.where(keep_mask)[0])
    return clipped_pcd


    

def create_pcd_from_rgbd(depth: np.ndarray, color: np.ndarray) -> o3d.geometry.PointCloud:
    u, v = np.meshgrid(np.arange(Config.WIDTH), np.arange(Config.HEIGHT))
    
    x = (u - Config.CX) * depth / Config.FX_STEREO
    y = (v - Config.CY) * depth / Config.FY_STEREO
    z = depth 
    
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    colors = color.reshape(-1, 3) / 255.0
   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def fit_circle(xs, ys):
    A = np.c_[2*xs, 2*ys, np.ones(xs.shape[0])]
    b = xs**2 + ys**2

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, c = sol
    r = np.sqrt(c + xc**2 + yc**2)
    return xc, yc, r