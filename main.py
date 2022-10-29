import cv2
import os
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
from PIL import Image

sometimes = lambda aug: iaa.Sometimes(0.4,aug) #40%만 적용
seq = iaa.Sequential([
    iaa.Fliplr(0.5), #사진 180도 회전 -> 50%수평 뒤집기
    iaa.Flipud(0.3), # vertically flip 30% of all images
    sometimes(iaa.GaussianBlur(sigma=(0,2.0))), #가우시안블러
    sometimes(iaa.Crop(percent=(0, 0.1))),  # 이미지 너비나 높이 0~10% 몇장 잘라내기
    sometimes(iaa.Superpixels(
                #Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image,
                        p_replace=(0, 0.2), # 픽셀의 20%정도만 변경
                        n_segments=(20, 200)
    )),
    sometimes(iaa.Sharpen(alpha=(0, 0.3), lightness=(0.80, 1.5))), #밝기 조절
    iaa.Emboss(alpha=(0, 0.5), strength=(0, 5.0)), #엠보싱 효과
    iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.3), direction=(0.0, 0.8)
                    ),
    ]), #검은색 마킹 효과
    sometimes(iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
    ])), #픽셀 일부 삭제
])
#a list of 3D numpy arrays, each having shape (height, width, channels).
# Grayscale images must have shape (height, width, 1) each.
# All images must have numpy's dtype uint8

path = './data/'
imgs =os.listdir(path)
jpg = [i for i in imgs if i.endswith(".jpg")]
image_np_list = []
image_name_list = []
image_np_dic = dict()
for x in jpg:
    img = Image.open(path+x)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    image_np_list.append(img_array)
    image_name_list.append(x.replace('.jpg',''))

k = 1
for i in range(20):
    change = seq(images=image_np_list)
    j = 0
    for x in change:
        cv2.imwrite('./result/{}_{}.jpg'.format(image_name_list[j],k),x)
        j += 1
    k+=1
