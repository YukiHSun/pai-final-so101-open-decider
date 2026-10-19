"""RGB-only red-cube localization on a calibrated, known flat table.

No simulator body IDs, depth buffer, segmentation IDs, or object poses enter
this module. Object size and table height are scene calibration, not live truth.
"""
from dataclasses import dataclass

import cv2
import numpy as np


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraCalibration:
    position: np.ndarray
    rotation: np.ndarray
    fovy_degrees: float


def locate_red_cube(image, camera: CameraCalibration, cube_edge=0.030, table_z=0.0):
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise VisionError("Expected uint8 HWC RGB")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) &
            (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 70)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 20]
    if len(candidates) != 1:
        raise VisionError(f"Expected one visible red cube; found {len(candidates)} candidates")
    i = candidates[0]
    x, y, w, h, area = stats[i]
    if x == 0 or y == 0 or x+w >= rgb.shape[1] or y+h >= rgb.shape[0]:
        raise VisionError("Red object is clipped by the camera")
    if not 0.65 < w / h < 1.5 or area / (w*h) < 0.7:
        raise VisionError("Red cube is occluded or not approximately square")
    u, v = centroids[i]
    height, width = rgb.shape[:2]
    f = 0.5 * height / np.tan(np.deg2rad(camera.fovy_degrees) / 2)
    ray = camera.rotation @ np.array([(u+0.5-width/2)/f, -(v+0.5-height/2)/f, -1.0])
    if abs(ray[2]) < 1e-6:
        raise VisionError("Camera ray is parallel to the table")
    distance = (table_z + cube_edge - camera.position[2]) / ray[2]
    if distance <= 0:
        raise VisionError("Table intersection is behind the camera")
    surface = camera.position + distance*ray
    center = np.array([surface[0], surface[1], table_z+cube_edge/2])
    if not (0.18 <= center[0] <= 0.26 and abs(center[1]) <= 0.04):
        raise VisionError("Red cube is outside the calibrated grasp workspace")
    return center
