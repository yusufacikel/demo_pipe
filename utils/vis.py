import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self):
        pass

    def gray(self, gray_image):
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        plt.show()

    def depth_global(self, depth_image):
        depth_image_nan = depth_image.copy().astype(np.float32)
        depth_image_nan[depth_image_nan == 0] = np.nan

        global_min = 0.0
        global_max = 500.0

        normalized = (depth_image_nan - global_min) / (global_max - global_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        normalized = normalized * 255

        img_normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)

        nan_mask = np.isnan(depth_image_nan)
        img_colored[nan_mask] = [0, 0, 0]

        plt.imshow(img_colored)
        plt.axis('off')
        plt.show()

    def depth_local(self, depth_image):
        depth_image_nan = depth_image.copy().astype(np.float32)
        depth_image_nan[depth_image_nan == 0] = np.nan

        global_min = np.nanmin(depth_image_nan)
        global_max = np.nanmax(depth_image_nan)

        normalized = (depth_image_nan - global_min) / (global_max - global_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        normalized = normalized * 255

        img_normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)

        nan_mask = np.isnan(depth_image_nan)
        img_colored[nan_mask] = [0, 0, 0]

        plt.imshow(img_colored)
        plt.axis('off')
        plt.show()
    
    def color(self, color_image):
        if np.any(np.isnan(color_image)):
            color_image = np.nan_to_num(color_image, nan=0)
        
        if color_image.dtype != np.uint8:
            color_image = color_image.astype(np.uint8)
        
        if len(color_image.shape) == 3 and color_image.shape[2] == 3:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        plt.imshow(color_image)
        plt.axis('off')
        plt.show()

    def pointcloud(self, pcd):    
        vis = o3d.visualization.Visualizer() # type: ignore
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])
        view_ctl = vis.get_view_control()
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = np.eye(4)
        view_ctl.convert_from_pinhole_camera_parameters(cam_params)
        vis.run()
        vis.destroy_window()

    def geoset(self, geometries):
        vis = o3d.visualization.Visualizer() # type: ignore
        vis.create_window()
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])
        view_ctl = vis.get_view_control()
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()
        extrinsic = np.eye(4)
        left_camera = [18,0,0]
        extrinsic[:3, 3] = left_camera
        cam_params.extrinsic = extrinsic
        view_ctl.convert_from_pinhole_camera_parameters(cam_params)

        vis.run()
        vis.destroy_window()

    