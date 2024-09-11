import torchvision.transforms as T
from PIL import Image
import random
import math
import cv2
import numpy as np
def random_affine_seq(img, hflip=0.5, degrees=(-0, 0), translate=(.2, .2), scale=(0.6, 1.4), shear=(-0, 0),
                  borderValue=(128, 128, 128)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # flipping
    F = np.eye(3)
    hflip = np.random.rand() < hflip
    if hflip:
        F[0, 0] = -1
        F[0, 2] = width

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * width + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * height + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R @ F  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue



    return imw, M
img_shape = [1080,1920]

t = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(img_shape) * 8 // 12).tolist())])
img = np.array(Image.open('/root/autodl-tmp/MultiviewX/Image_subsets/C1/0000.png').convert('RGB'))
img, M = random_affine_seq(img)
img = t(img)
b = 1
pass