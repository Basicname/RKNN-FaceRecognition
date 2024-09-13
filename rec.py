import cv2
import numpy as np
from rknnlite.api import RKNNLite

rknn = RKNNLite()

def init():
    rknn.load_rknn('mobilefacenet.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

def get_feat(img):
    img = img[..., ::-1]
    blob = np.expand_dims(img, axis=0)
    net_out = rknn.inference(inputs=[blob])[0]
    return net_out

