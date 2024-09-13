import cv2
import numpy as np

def affineMatrix(nose, leftEyeCenter, rightEyeCenter, scale=2.5):
    nose = np.array(nose, dtype=np.float32)
    left_eye = np.array(leftEyeCenter, dtype=np.float32)
    right_eye = np.array(rightEyeCenter, dtype=np.float32)
    eye_width = right_eye - left_eye
    angle = np.arctan2(eye_width[1], eye_width[0])
    center = nose
    alpha = np.cos(angle)
    beta = np.sin(angle)
    w = np.sqrt(np.sum(eye_width**2)) * scale
    m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
        [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
    return np.array(m), (int(w), int(w))

def align(img, nose, leftEyeCenter, rightEyeCenter):
    mat, size = affineMatrix(nose, leftEyeCenter, rightEyeCenter)
    img = cv2.warpAffine(img, mat, size)
    img = cv2.resize(img, (112, 112))
    return img
