import cv2 as cv
import math
import numpy as np

f = cv.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\lena512.jpg")
h, w = f.shape[:2]
f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)

m_g = np.zeros((h, w))
a_g = np.zeros((h, w))
aa_g = np.zeros((h, w))
nms_g = np.zeros((h, w))


blur = cv.GaussianBlur(f, (7, 7), 0)
f = blur

gy = cv.Sobel(f, cv.CV_64F, 1, 0, ksize=3)
gx = cv.Sobel(f, cv.CV_64F, 0, 1, ksize=3)

for x in range(h):
    for y in range(w):
        m_g[x, y] = (math.sqrt((pow(gx[x, y], 2)+pow(gy[x, y], 2))))

for x in range(h):
    for y in range(w):
        a_g[x, y] = np.arctan2(gy[x, y], gx[x, y])*180/math.pi
        if (a_g[x, y] < 0):
            a_g[x, y] = a_g[x, y] + 180
        a_g[x, y] = int(a_g[x, y]/45)


for x in range(1, h-1):
    for y in range(1, w-1):
        if (a_g[x, y] == 0 and m_g[x, y] > m_g[x - 1, y] and m_g[x, y] > m_g[x + 1, y]):
            nms_g[x, y] = m_g[x, y]
        if (a_g[x, y] == 1 and m_g[x, y] > m_g[x + 1, y + 1] and m_g[x, y] > m_g[x - 1, y - 1]):
            nms_g[x, y] = m_g[x, y]
        if (a_g[x, y] == 2 and m_g[x, y] > m_g[x, y + 1] and m_g[x, y] > m_g[x, y - 1]):
            nms_g[x, y] = m_g[x, y]
        if (a_g[x, y] == 3 and m_g[x, y] > m_g[x + 1, y - 1] and m_g[x, y] > m_g[x - 1, y + 1]):
            nms_g[x, y] = m_g[x, y]

def thr(f, a, b):
    canny = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            if(f[x, y]>a):
                canny[x, y] = 255
            elif(f[x, y]>b):
                canny[x, y] = 128
    return canny

canny_edge = thr(nms_g,30, 20)
cv.imwrite('gx.jpg', gx)
cv.imwrite('gy.jpg', gy)
cv.imwrite('nms.jpg', nms_g)
cv.imwrite('canny2010.jpg', canny_edge)