# #!/usr/bin/env python
import numpy as np
import string
import os
from math import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("lenna.png", 0)
# image = cv2.resize(image, [512,512])
image = cv2.resize(image, [100,100])
img = 255*np.ones((104,104))
img[2:102,2:102] = image

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

codes = dict()

# Using OpenCV
def dct_cv2(img): # B: blocks - 8
    w,h = np.shape(img)
    # vis0 = np.zeros((h,w), np.float32)
    dct_img = np.zeros((h,w), np.float32)

    # vis0[:h,:w] = img
    # vis0 = np.float32(img)/255.0 # Scaled
    img = np.float32(img)/255.0 # Scaled
    h,w = [int(h/8), int(w/8)]

    for row in range(h):
        for col in range(w):
            cur_block = cv2.dct(img[row*8:(row+1)*8, col*8:(col+1)*8])
            dct_img[row*8:(row+1)*8, col*8:(col+1)*8] = cur_block

    dct_img = dct_img*255.0
    # dct_img = np.uint8(dct_img) # convert back to int

    return dct_img

def quantization(dct_img,Q):
    w,h = np.shape(dct_img)
    q_img = np.zeros_like(dct_img)
    h,w = [int(h/8), int(w/8)]

    for row in range(h):
        for col in range(w):
            cur_block = dct_img[row*8:(row+1)*8, col*8:(col+1)*8]
            q_block = np.zeros_like(cur_block)
            for i in range(8):
                for j in range(8):
                    q_block[i,j] = round(cur_block[i,j] / Q[i,j])

            q_img[row*8:(row+1)*8, col*8:(col+1)*8] = q_block

    return q_img

# def dialogal(n, i, N):
#     if i == 1:
#         if n != 0:
#             dialogal(n-1, i, N)
#         re = []
#         i = 0; j = n
#         re.append([i,j])
#         if j != 0:
#             while j != 0:
#                 i += 1; j -= 1
#                 re.append([i,j])
#     else:
#         if n != 1:
#             dialogal(n-1, i, N)
#         re = []
#         i = N; j = n
#         re.append([i,j])
#         if j != N:
#             while j != N:
#                 i -= 1; j += 1
#                 re.append([i,j])
#     if len(re) % 2 != 0:
#         re.reverse()
#     print(re)

def zig_zag(input_matrix, block_size):
    z = np.empty([block_size*block_size])
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
                z[index] = input_matrix[j, i-j]
            else:
                z[index] = input_matrix[i-j, j]
    return z

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

def create_tree(z):
    dict_data = Frequency(z)
    symbols = dict_data.keys()
    freq = dict_data.values()

    nodes = []
    
    # converting symbols and freq into huffman tree nodes
    for symbol in symbols:
        node = Node(dict_data.get(symbol), symbol)
        nodes.append(node)

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
    
        # pick 2 smallest nodes
        left = nodes[0]
        right = nodes[1]
    
        left.code = 0
        right.code = 1
    
        # combine the 2 smallest nodes to create new node
        root = Node(left.freq+right.freq, left.symbol+right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(root)

    return nodes

def huffman_encoding(z):
    nodes = create_tree(z)
    huffman = encoding(nodes[0])
    # print("symbols with codes", huffman)
    return huffman

def encoding(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if node.left:
        encoding(node.left, newVal)
    if node.right:
        encoding(node.right, newVal)

    if not node.left and not node.right:
        codes[node.symbol] = newVal
         
    return codes   

def Frequency(z):
    code = dict()
    for element in z:
        if code.get(element) == None:
            code[element] = 1
        else:
            code[element] += 1

    # code = sorted(code.items(), key=lambda x: x[1])
    # code = dict((x, y) for x, y in code)
    return code



def idct_cv2(dct_img):
    w, h = np.shape(dct_img)
    img = np.zeros((h,w), np.float32)
    dct_img = np.float32(dct_img)/255.0 # Scaled
    h,w = [int(h/8), int(w/8)]

    for row in range(h):
        for col in range(w):
            cur_block = cv2.idct(dct_img[row*8:(row+1)*8, col*8:(col+1)*8])
            img[row*8:(row+1)*8, col*8:(col+1)*8] = cur_block

    img = img*255.0 # convert back to int
    return img

if __name__ == '__main__':
    dct_img = dct_cv2(img)
    q_img = quantization(dct_img,Q_2)
    z = zig_zag(q_img,8)
    huffman = huffman_encoding(z)

    # img = idct_cv2(dct_img)

    plt.imshow(q_img)
    print(z)
    # plt.show()