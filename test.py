import get_embedding
import cv2
import sys

if len(sys.argv) != 3:
    print('Usage: python3 test.py source_img1 source_img2')
    exit()
img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])
get_embedding.init()
feature1 = get_embedding.get_embeddings(img1)[0]['embedding']
feature2 = get_embedding.get_embeddings(img2)[0]['embedding']
cosine_similarity = get_embedding.cosine_similarity(feature1, feature2)
print(cosine_similarity)

