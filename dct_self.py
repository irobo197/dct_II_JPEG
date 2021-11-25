# #!/usr/bin/env python
import numpy as np
from numpy import random
from math import cos, sin, sqrt, pi
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("lenna.png", 0)
img = cv2.resize(img, [100,100])
image = 255*np.ones((104,104))
image[2:102,2:102] = img

# image = random.randint(10, size=(16, 16))

# DCT-II-1D
# def transform(vector):
#     result = []
#     factor = math.pi / len(vector)
#     for i in range(len(vector)):
#         sum = 0.0
#         for (j, val) in enumerate(vector):
#             sum += val * math.cos((j + 0.5) * i * factor)
#         result.append(sum)
#     return result

# IDCT-II-1D == 2/N * DCT-III-1D
# def inverse_transform(vector):
#     result = []
#     factor = math.pi / len(vector)
#     for i in range(len(vector)):
#         sum = vector[0] / 2
#         for j in range(1, len(vector)):
#             sum += vector[j] * math.cos(j * (i + 0.5) * factor)
#         result.append(sum)
#     return result

# Origin DCT-Type II
def dctTransform(img):
    w, h = np.shape(img)
    dct = np.zeros_like(img)
    for i in range(h):
        for  j in range(w): 
            if i == 0:
                ci = 1 / sqrt(h)
            else:
                ci = sqrt(2) / sqrt(h)

            if j == 0:
                cj = 1 / sqrt(w)
            else:
                cj = sqrt(2) / sqrt(w)

            sum = 0
            for row in range(h):
                for col in range(w):
                    dct1 = img[row][col] * cos((2*row + 1)*i*pi / (2*h)) * cos((2*col + 1)*j*pi / (2*w))
                    sum = sum + dct1
                
            dct[i][j] = sum * 2/h
    return dct

def idctTransform(img):
    w, h = np.shape(img)
    dct = np.zeros_like(img)
    for i in range(h):
        for  j in range(w): 
            sum = 0
            for row in range(h):
                for col in range(w): 
                    if row == 0:
                        ci = 1 / sqrt(h)
                    else:
                        ci = sqrt(2) / sqrt(h)

                    if col == 0:
                        cj = 1 / sqrt(w)
                    else:
                        cj = sqrt(2) / sqrt(w)

                    dct1 = ci * cj * img[row][col] * cos((2*i + 1)*row*pi / (2*h)) * cos((2*j + 1)*col*pi / (2*w))
                    sum = sum + dct1
                
            dct[i][j] = sum * 2/h
    return dct

def dct_8(img):
    w, h = np.shape(img)
    dct_img = np.zeros((h,w), np.float32)
    # img = img - 128.0
    img = img/255.0 # Scaled
    w, h = [int(w/8), int(h/8)]

    for i in range(h):
        for j in range(w):
            temp = img[8*i:8*(i+1),8*j:8*(j+1)]
            result = dctTransform(temp)

            dct_img[8*i:8*(i+1),8*j:8*(j+1)] = result

    dct_img = dct_img*255.0
    # print(dct_img[0:8,0:8])
    # dct_img = np.float32(dct_img)
    # print(dct_img[0:8,0:8])
    # dct_img = dct_img + 128.0
    return dct_img

def idct_8(dct_img):
    w, h = np.shape(dct_img)
    img = np.zeros((h,w), np.float32)
    # img = img - 128.0
    img = dct_img/255.0 # Scaled
    w, h = [int(w/8), int(h/8)]

    for i in range(h):
        for j in range(w):
            temp = dct_img[8*i:8*(i+1),8*j:8*(j+1)]
            result = idctTransform(temp)

            img[8*i:8*(i+1),8*j:8*(j+1)] = result

    img = img*255.0
    # print(img[0:8,0:8])
    # dct_img = dct_img + 128.0
    return img

# Using OpenCV
def dct_cv2(img, B): # B: blocks - 8
    w,h = np.shape(img)
    # vis0 = np.zeros((h,w), np.float32)
    dct_img = np.zeros((h,w), np.float32)

    # vis0[:h,:w] = img
    # vis0 = np.float32(img)/255.0 # Scaled
    img = np.float32(img)/255.0 # Scaled
    h,w = [int(h/B), int(w/B)]

    for row in range(h):
        for col in range(w):
            cur_block = cv2.dct(img[row*B:(row+1)*B, col*B:(col+1)*B])
            dct_img[row*B:(row+1)*B, col*B:(col+1)*B] = cur_block

    dct_img = dct_img*255.0
    # dct_img = np.uint8(dct_img) # convert back to int

    return dct_img

def idct_cv2(dct_img,B):
    w, h = np.shape(dct_img)
    img = np.zeros((h,w), np.float32)
    dct_img = np.float32(dct_img)/255.0 # Scaled
    h,w = [int(h/B), int(w/B)]

    for row in range(h):
        for col in range(w):
            cur_block = cv2.idct(dct_img[row*B:(row+1)*B, col*B:(col+1)*B])
            img[row*B:(row+1)*B, col*B:(col+1)*B] = cur_block

    img = img*255.0 # convert back to int
    return img

def dct_8_4(dct_img):
    w, h = np.shape(dct_img)
    dct_img_4 = np.zeros_like(dct_img)
    h,w = [int(h/8), int(w/8)]

    for row in range(h):
        for col in range(w):
            cur_block = dct_img[row*8:(row+1)*8, col*8:(col+1)*8]
            for i in range(4,8):
                for j in range(4,8):
                    cur_block[i,j] = 0.0

            dct_img_4[row*8:(row+1)*8, col*8:(col+1)*8] = cur_block

    return dct_img_4

if __name__ == '__main__':

    ''' Cau a,b '''
    dct_img = dct_8(image)
    # plt.imshow(dct_img)
    # plt.show()

    # dct_img = dct_cv2(image, 8)
    # b=dct_img[0:8,0:8]
    # plt.imshow(a-b)
    # plt.show()
    ''' Cau c '''
    dct_img_4 = dct_8_4(dct_img)
    # plt.imshow(dct_img_4)
    img = idct_8(dct_img_4)

    # img = idct_cv2(dct_img,8)
    plt.imshow(img)
    plt.show()