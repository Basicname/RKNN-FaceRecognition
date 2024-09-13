import numpy as np
import cv2
from math import ceil
from itertools import product as product
import face_align
import time

from rknnlite.api import RKNNLite

rknn = RKNNLite(verbose=False)

def init():
    rknn.load_rknn('RetinaFace_resnet50_320.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

def PriorBox(image_size): #image_size Support (320,320) and (640,640)
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors).reshape(-1, 4)
    return output

def box_decode(loc, priors):
    variances = [0.1, 0.2]
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors):
    variances = [0.1, 0.2]
    landmarks = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
    ), axis=1)
    return landmarks

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def get_faces(img):
    img_height, img_width, _ = img.shape
    model_height, model_width = (320, 320)
    letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, (model_height,model_width), 114)  # letterbox缩放
    infer_img = letterbox_img[..., ::-1] 

    i=np.expand_dims(infer_img,0)
    start_time = time.time()
    outputs = rknn.inference(inputs=[i])
    if outputs == None: return None
    loc, conf, landmarks = outputs
    priors = PriorBox(image_size=(model_height, model_width))
    boxes = box_decode(loc.squeeze(0), priors)
    scale = np.array([model_width, model_height,
                      model_width, model_height])
    boxes = boxes * scale // 1 
    boxes[...,0::2] =np.clip((boxes[...,0::2] - offset_x) / aspect_ratio, 0, img_width)  #letterbox
    boxes[...,1::2] =np.clip((boxes[...,1::2] - offset_y) / aspect_ratio, 0, img_height) #letterbox
    scores = conf.squeeze(0)[:, 1] 
    landmarks = decode_landm(landmarks.squeeze(
        0), priors) 
    scale_landmarks = np.array([model_width, model_height, model_width, model_height,
                                model_width, model_height, model_width, model_height,
                                model_width, model_height])
    landmarks = landmarks * scale_landmarks // 1
    landmarks[...,0::2] = np.clip((landmarks[...,0::2] - offset_x) / aspect_ratio, 0, img_width) #letterbox
    landmarks[...,1::2] = np.clip((landmarks[...,1::2] - offset_y) / aspect_ratio, 0, img_height) #letterbox
    
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = nms(dets, 0.5)
    dets = dets[keep, :]
    landmarks = landmarks[keep]
    dets = np.concatenate((dets, landmarks), axis=1)
    ret = []
    for data in dets:
        if float(data[4]) < 0.6:
            continue
        x1 = int(data[0])
        y1 = int(data[1])
        x2 = int(data[2])
        y2 = int(data[3])
        x3 = int(data[5])
        y3 = int(data[6])
        x4 = int(data[7])
        y4 = int(data[8])
        x5 = int(data[9])
        y5 = int(data[10])
        
        leftEyeCenter = np.array([x3, y3])
        rightEyeCenter = np.array([x4, y4])
        nose = np.array([x5, y5])
        
        face_aligned = face_align.align(img, nose, leftEyeCenter, rightEyeCenter)
        faces = {'face' : face_aligned, 'score' : data[4], 'point1' : x1, 'point2': y1 + 12}
        ret.append(faces)
    # Release
    return ret
