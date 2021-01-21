import cv2 as cv
import math
import numpy as np

# f = cv.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\lena512.bmp")
f = cv.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\test.bmp")
h, w = f.shape[:2]
f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)

I_x = np.zeros((h, w))
I_y = np.zeros((h, w))
I_xy = np.zeros((h, w))
r = np.zeros((h, w))

blur = cv.GaussianBlur(f, (7, 7), 0)
f = blur

gx = cv.Sobel(f, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(f, cv.CV_64F, 0, 1, ksize=3)

for x in range(h):
    for y in range(w):
        I_x[x, y] = (pow(gx[x, y], 2))
        I_y[x, y] = (pow(gy[x, y], 2))
        I_xy[x, y] = (gx[x, y]*gy[x, y])

nn = 4
# nn = 3
# nn = 5
o = 25

# o = 9
i_x_2 = cv.GaussianBlur(I_x, (o, o), nn)
i_y_2 = cv.GaussianBlur(I_y, (o, o), nn)
i_xy = cv.GaussianBlur(I_xy, (o, o), nn)

for x in range(h):
    for y in range(w):
        m = np.zeros((2, 2))
        m[0, 0] = i_x_2[x, y]
        m[0, 1] = i_xy[x, y]
        m[1, 0] = i_xy[x, y]
        m[1, 1] = i_y_2[x, y]

        r[x, y] = (m[0, 0]*m[1, 1] - pow(m[1, 0], 2) - 0.06*pow((m[1, 1] + m[0, 0]), 2))

def thr(f, a):
    harris = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            if(f[x, y]>a):
                harris[x, y] = 255
    return harris

img = thr(r, 0)
cv.imwrite('harris.jpg', img)
cv.imwrite('r.jpg', r)
cv.imwrite('gx.jpg', gx)
cv.imwrite('gy.jpg', gy)
cv.imwrite('ix.jpg', i_x_2)
cv.imwrite('iy.jpg', i_y_2)
cv.imwrite('ixy.jpg', i_xy)


