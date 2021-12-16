#!/usr/bin/env python
import numpy as np
import time
from threading import Thread
from math import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import cv2
from huffman_coding import huffman_encoding, huffman_decoding
from utlis import *

Q_1 = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                [12, 12, 14, 19,  26,  58,  60,  55],
                [14, 13, 16, 24,  40,  57,  69,  56],
                [14, 17, 22, 29,  51,  87,  80,  62],
                [18, 22, 37, 56,  68, 109, 103,  77],
                [24, 35, 55, 64,  81, 104, 113,  92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103,  99]])

Q_2 = np.array([[ 8,  6,  5,  8,  12,  20,  26,  31],
                [ 6,  6,  7, 10,  13,  29,  30,  28],
                [ 7,  7,  8, 12,  20,  29,  35,  28],
                [ 7,  9, 11, 15,  26,  44,  40,  31],
                [ 9, 11, 19, 28,  34,  55,  52,  39],
                [12, 18, 28, 32,  41,  52,  57,  46],
                [25, 32, 39, 44,  52,  61,  60,  51],
                [36, 46, 48, 49,  56,  50,  52,  50]])

Q_3 = np.array([[ 24,  18,  15,  24,  36,  60,  78,  93],
                [ 18,  18,  21,  30,  39,  87,  90,  84],
                [ 21,  21,  24,  36,  60,  87, 105,  84],
                [ 21,  27,  33,  45,  78, 132, 120,  93],
                [ 27,  33,  57,  84, 102, 165, 132, 117],
                [ 36,  54,  84,  96, 123, 156, 171, 138],
                [ 75,  96, 117, 132, 156, 183, 180, 153],
                [108, 138, 144, 147, 168, 150, 156, 150]])

# Quantization
def quantization(dct_block,Q = Q_1):
    q_block = np.zeros_like(dct_block)
    for i in range(8):
        for j in range(8):
            q_block[i,j] = round(dct_block[i,j] / Q[i,j])

    return q_block

# Zigzag
def zig_zag(q_block, block_size = 8):
    z = np.zeros([block_size*block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = q_block[j, i-j]
            else:
                z[index] = q_block[i-j, j]
    return z

# Inverse zigzag
def zig_zag_reverse(z, block_size = 8):
    q_block = np.zeros([block_size, block_size])
    index = -1
    bound = 0
    input_m = []
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            #print(len(z))
            if len(z) != 0:
                if i % 2 == 1:
                    q_block[j, i - j] = z[index]
                else:
                    q_block[i - j, j] = z[index]
            else:
                if i % 2 == 1:
                    q_block[j, i - j] = 0.0
                else:
                    q_block[i - j, j] = 0.0

    return q_block

# Inverse quantization
def iquantization(q_block, Q = Q_1):
    dct_block = np.zeros_like(q_block)
    for i in range(8):
        for j in range(8):
            dct_block[i,j] = q_block[i,j] * Q[i,j]

    return dct_block

class JPEG(Thread):
    def __init__(self, image):
        super(JPEG, self).__init__()
        self.image = image
        self.compressed_image = np.zeros_like(image)

    def run(self):
        img, h_up, h_down, w_left, w_right = resize_img(image)
        img = np.float32(img)/255.0 # Scaled

        h,w = np.shape(img)
        h_normal, w_normal = [int(h/8), int(w/8)]
        compressed_img = np.zeros_like(img)

        for row in range(h_normal):
            for col in range(w_normal):
                dct_block = cv2.dct(img[row*8:(row+1)*8, col*8:(col+1)*8])
                dct_block = dct_block * 255.0

                q_block = quantization(dct_block)

                z = zig_zag(q_block)
                #print(np.shape(q_block),z)
                huffman, tree = huffman_encoding(z)

                z = huffman_decoding(huffman, tree)
                q_block = zig_zag_reverse(z)
            
                dct_block = iquantization(q_block)

                idct_block = cv2.idct(dct_block)
                compressed_img[row*8:(row+1)*8, col*8:(col+1)*8] = idct_block

        compressed_image = compressed_img[h_up:h - h_down, w_left: w - w_right]
        return compressed_image

if __name__ == '__main__':
    t = time.time()

    image = cv2.imread("lenna.png")
    #image = cv2.resize(image, [100,100])
    #compressed_image = run(image)

    b,g,r = cv2.split(image)
    compressed_b = threading.Thread(target=run, args=(b,))
    compressed_g = threading.Thread(target=run, args=(g,))
    compressed_r = threading.Thread(target=run, args=(r,))

    compressed_b.start()
    compressed_g.start()
    compressed_r.start()

    # Wait for another thread done???
    compressed_b.join()
    compressed_g.join()
    compressed_r.join()   

    print(type(compressed_b))
    #compressed_image = cv2.merge([compressed_r, compressed_g, compressed_b])
    #compressed_image = np.uint8(compressed_image)
    #print(compressed_image)
    #mse = MSE(compressed_image, image)
    #psnr = PSNR(mse)
    #print("PSNR:",psnr)

    #plt.subplot(121)
    #plt.imshow(image)
    #plt.title("Original")

    #plt.subplot(122)
    #plt.imshow(compressed_b)
    #plt.title("Compressed")
    #plt.show()

    #plt.imsave("output.jpg", compressed_image)
    print('done in ',time.time() - t)