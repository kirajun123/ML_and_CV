# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
'''
对图像进行处理，包括：
图像剪裁函数image_crop
图像颜色随机变换函数random_light_color
图像伽马校正函数adjust_gamma
图像旋转函数img_rotation
图像随机投影变换函数random_warp
'''

#image crop 图像剪裁
def image_crop(img, height, width):
    '''
    对图像进行剪裁
    height,width为包含2个元素的list，表示需要剪裁的起点、终点
    '''
    return img[height[0]:height[1], width[0]:width[1]]    
    
#color shift
#利用随机数产生器，对图片颜色随机进行改变
def random_light_color(img):
    B, G, R = cv2.split(img)
    #在-50到50之间产生随机整数
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    #由于B、G、R通道中，代表颜色的整数均在0~255范围内，因此要进行一定处理，以防止数字超出范围
    elif b_rand > 0:
        #计算限值
        lim = 255 - b_rand
        #若加了该随机数后会超出255的B中元素，直接另其达到最大值255
        B[B>lim] = 255
        #其他不会超出255的元素，直接加该随机数
        B[B<lim] = (b_rand + B[B<lim]).astype(img.dtype)
    elif b_rand < 0:
        #计算限值
        lim = 0 - b_rand
        #加了该随机数后会小于0的B中元素，直接另其达到最小值0
        B[B < lim] = 0
        #其他不会低于0的元素，直接加该随机数
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    #对G、R作同样处理
    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    #将增加了随机数后的BGR数组组合起来，形成新的图片
    img_merge = cv2.merge((B, G, R))
    #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge

#gamma correction 伽马校正
def adjust_gamma(img, gamma = 1.0):
    #build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invgamma = 1/gamma
    table = []
    for i in range(256):
        table.append(((i/255.0)**invgamma)*255)
    #将table list转为np数组
    table = np.array(table).astype("uint8")
    #The function LUT fills the output array with values from the look-up table
    return cv2.LUT(img, table)

#rotation旋转
def img_rotation(img, center, angle = 30, scale = 1.0):
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate

#perspective transform 投影变换
def random_warp(img, row, col):
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    
    return M_warp, img_warp

if __name__ == '__main__':
    img = cv2.imread('dark.jpg')
    height = [0, 100]
    width = [2, 102]
    img_crop = image_crop(img, height, width)
    cv2.imshow('lenna_crop', img_crop)