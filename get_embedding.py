import argparse
import retinaface
import rec
import cv2
import numpy as np
import time

def get_embeddings(image):
    retina_ret = retinaface.get_faces(image)
    if retina_ret == None: return None
    embedder_ret = []
    for face in retina_ret:
        embedding = rec.get_feat(face['face'])
        embedder_ret.append({'score':face['score'], 'embedding':embedding})
    return embedder_ret

def cosine_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)

    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity
    
def compute_sim(embedding1, embedding2, thres=0.635):
    distance = cosine_similarity(embedding1, embedding2)
    return distance > thres

def compare_face(face_path1, face_path2):
    face1 = cv2.imread(face_path1)
    feat1 = get_embeddings(face1)
    feat1 = feat1[0]['embedding']
    return compute_sim(feat1, feat1)

def init():
    retinaface.init()
    rec.init()
    
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Mobilefacenet RKNN Demo', add_help=True)
    parser.add_argument('--source', type=str, help='source image')
    args = parser.parse_args()
    init()
    print(f'Initialize costs : {time.time() - start_time}s')
    img = cv2.imread(args.source)
    start_time = time.time()
    get_embeddings(img)
    end_time = time.time()
    print(f'Inference costs : {end_time - start_time}s')
