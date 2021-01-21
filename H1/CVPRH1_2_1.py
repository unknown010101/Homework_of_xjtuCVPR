import cv2 as cv
import math
import numpy as np

f = cv.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\lena512.jpg")
h, w = f.shape[:2]
t = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
level = 5

n = 0

m_g = np.zeros((h, w))
a_g = np.zeros((h, w))
hvs_g = cv.cvtColor(f, cv.COLOR_RGB2HSV)

blur = cv.GaussianBlur(t, (3, 3), 4*n)
t = blur

gx = cv.Sobel(t, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(t, cv.CV_64F, 0, 1, ksize=3)

def s(f):
    if(f>=255):
        f = 255
    elif(f < 0):
        f = 0
    return f


for x in range(h):
    for y in range(w):
        m_g[x, y] = s(math.sqrt((pow(gx[x, y], 2)+pow(gy[x, y], 2))))
        a_g[x, y] = np.arctan2(gy[x, y], gx[x, y])

for x in range(h):
    for y in range(w):
        hvs_g[x, y, 1] = int((m_g[x, y]))
        hvs_g[x, y, 0] = ((a_g[x, y]*180)/math.pi)/2
        if (a_g[x, y]<0):
            hvs_g[x, y, 0] = (hvs_g[x, y, 0] + 180)
        hvs_g[x, y, 2] = 255

rgb_g = cv.cvtColor(hvs_g, cv.COLOR_HSV2RGB)
cv.imwrite('M_' + str(n) + '.jpg', m_g)
# cv.imwrite('A_' + str(n) + '.jpg', rgb_g)