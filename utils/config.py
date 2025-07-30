# utils/config.py
import numpy as np
import os

class Config:
    PIPE_DEMO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PIPE_DEMO_ROOT, "sample_data/run_03_syn")    
    FRAME_RATE = 30.0
    WIDTH = 848
    HEIGHT = 480
    FX = 422.068
    FY = 424.824
    CX = 404.892
    CY = 260.621
    INTRINSICS = np.array([[FX, 0, CX],
                          [0, FY, CY],
                          [0, 0, 1]], dtype=np.float32)
    STEREO_FACTOR = 2.0
    FX_STEREO = FX * STEREO_FACTOR
    FY_STEREO = FY * STEREO_FACTOR
    INTRINSICS_STEREO = np.array([[FX_STEREO, 0, CX],
                                 [0, FY_STEREO, CY],
                                 [0, 0, 1]], dtype=np.float32)
    CLIP_UPPER = 375.0
    CLIP_LOWER = 175.0