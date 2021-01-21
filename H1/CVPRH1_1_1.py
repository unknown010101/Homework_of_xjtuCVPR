import cv2 as cv
import math
import numpy as np

f = cv.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\lena512.bmp")

def t(x, y):
    maritx = np.zeros((3, 3))
    maritx[0, 0] = 1
    maritx[0, 1] = 0
    maritx[0, 2] = x
    maritx[1, 0] = 0
    maritx[1, 1] = 1
    maritx[1, 2] = y
    maritx[2, 0] = 0
    maritx[2, 1] = 0
    maritx[2, 2] = 1
    return maritx

def r(theta):
    maritx = np.zeros((3, 3))
    a = (theta*math.pi)/180
    maritx[0, 0] = math.cos(a)
    maritx[0, 1] = -math.sin(a)
    maritx[0, 2] = 0
    maritx[1, 0] = math.sin(a)
    maritx[1, 1] = math.cos(a)
    maritx[1, 2] = 0
    maritx[2, 0] = 0
    maritx[2, 1] = 0
    maritx[2, 2] = 1
    return maritx

def e(theta, x, y):
    maritx = np.zeros((3,3))
    a = (theta*math.pi)/180
    maritx[0, 0] = math.cos(a)
    maritx[0, 1] = -math.sin(a)
    maritx[0, 2] = x
    maritx[1, 0] = math.sin(a)
    maritx[1, 1] = math.cos(a)
    maritx[1, 2] = y
    maritx[2, 0] = 0
    maritx[2, 1] = 0
    maritx[2, 2] = 1
    return maritx

def s(theta, x, y, s):
    maritx = np.zeros((3,3))
    a = (theta*math.pi)/180
    maritx[0, 0] = s*math.cos(a)
    maritx[0, 1] = s*(-math.sin(a))
    maritx[0, 2] = x
    maritx[1, 0] = s*math.sin(a)
    maritx[1, 1] = s*math.cos(a)
    maritx[1, 2] = y
    maritx[2, 0] = 0
    maritx[2, 1] = 0
    maritx[2, 2] = 1
    return maritx

def a(x, y):
    maritx = np.zeros((3,3))
    maritx[0, 0] = 0.765*3
    maritx[0, 1] = -0.122*3
    maritx[0, 2] = x
    maritx[1, 0] = -0.174*3
    maritx[1, 1] = 0.916*3
    maritx[1, 2] = y
    maritx[2, 0] = 0
    maritx[2, 1] = 0
    maritx[2, 2] = 1
    return maritx

def z(h, w):
    maritx = np.zeros((3,3))
    src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float32)  # 原图像的4个需要变换的像素点（4个顶角）
    dst = np.array([[80, 80], [w / 2, 50], [80, h - 80], [w / 2, w / 2]], np.float32)  # 投影变换的4个像素点
    M = cv.getPerspectiveTransform(src, dst)
    maritx = M
    return maritx

def T(f, i, j):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            f_maritx = t(-i, -j)
            x = u*f_maritx[0, 0] + v*f_maritx[0, 1] + f_maritx[0, 2]
            y = u*f_maritx[1, 0] + v*f_maritx[1, 1] + f_maritx[1, 2]
            x = int(x)
            y = int(y)
            if (-1< x < h and -1< y < w):
                g[u, v] = f.item(x, y, 0)
    return g

def R(f, theta):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            f_maritx = r(theta)
            u = x*f_maritx[0, 0] + y*f_maritx[0, 1] + f_maritx[0, 2]
            v = x*f_maritx[1, 0] + y*f_maritx[1, 1] + f_maritx[1, 2]
            u = int(u)
            v = int(v)
            if (-1< u < h and -1< v < w):
                g[u, v] = f.item(x, y, 0)
    return g

def R_pro(f, theta):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            f_maritx = r(-theta)
            x = u*f_maritx[0, 0] + v*f_maritx[0, 1] + f_maritx[0, 2]
            y = u*f_maritx[1, 0] + v*f_maritx[1, 1] + f_maritx[1, 2]
            x = int(x)
            y = int(y)
            if (-1< x < h and -1< y < w):
                g[u, v] = f.item(x, y, 0)
    return g

def E(f, theta, i, j):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            f_maritx = e(-theta, -i, -j)
            x = u*f_maritx[0, 0] + v*f_maritx[0, 1] + f_maritx[0, 2]
            y = u*f_maritx[1, 0] + v*f_maritx[1, 1] + f_maritx[1, 2]
            x = int(x)
            y = int(y)
            if (-1< x < h and -1< y < w):
                g[u, v] = f.item(x, y, 0)
    return g

def S(f, theta, i, j, k):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            f_maritx = s(-theta, -i, -j, 1/k)
            x = u*f_maritx[0, 0] + v*f_maritx[0, 1] + f_maritx[0, 2]
            y = u*f_maritx[1, 0] + v*f_maritx[1, 1] + f_maritx[1, 2]
            x = int(x)
            y = int(y)
            if (-1< x < h and -1< y < w):
                g[u, v] = f.item(x, y, 0)
    return g

def A(f, i, j):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            f_maritx = a(i, j)
            x = u*f_maritx[0, 0] + v*f_maritx[0, 1] + f_maritx[0, 2]
            y = u*f_maritx[1, 0] + v*f_maritx[1, 1] + f_maritx[1, 2]
            x = int(x)
            y = int(y)
            if (-1< x < h and -1< y < w):
                g[u, v] = f.item(x, y, 0)
    return g

def Z(f):
    h, w, c = f.shape
    g = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            f_maritx = z(h, w)
            u = (x*f_maritx[0, 0] + y*f_maritx[0, 1] + f_maritx[0, 2])/(x*f_maritx[2, 0] + y*f_maritx[2, 1] + f_maritx[2, 2])
            v = (x*f_maritx[1, 0] + y*f_maritx[1, 1] + f_maritx[1, 2])/(x*f_maritx[2, 0] + y*f_maritx[2, 1] + f_maritx[2, 2])
            u = int(u)
            v = int(v)
            if (-1< u < h and -1< v < w):
                g[u, v] = f.item(x, y, 0)
    return g

# g_t = T(f, 128, 128)
# g_r = R(f, 45)
# g_r_pro = R_pro(f, 45)
# g_e = E(f, 10, 50, 10)
# g_s = S(f, 0, 0, 0, 2)
g_a = A(f, 10, 50)
# g_z = Z(f)

# cv.imshow('T', g_t)
# cv.imwrite('T.jpg', g_t)
#
# cv.imshow('R', g_r)
# cv.imwrite('R.jpg', g_r)
# cv.imshow('R_pro', g_r_pro)
# cv.imwrite('R_pro.jpg', g_r_pro)
#
# cv.imshow('E', g_e)
# cv.imwrite('E.jpg', g_e)

# cv.imshow('S', g_s)
# cv.imwrite('S.jpg', g_s)

cv.imshow('A', g_a)
cv.imwrite('A.jpg', g_a)
#
# cv.imshow('Z', g_z)
# cv.imwrite('Z.jpg', g_z)
