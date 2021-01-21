import cv2 as cv
import numpy as np

f = cv.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\lena512.bmp")
h, w = f.shape[:2]
level = 4
t = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
G_cell = np.empty((level+1, 1), dtype=object)
G_cell[0, 0] = t
cv.imwrite('G_pyramid_' + str(0) + '.jpg', f)

for n in range(level):
    i = int(h/pow(2, n+1))
    j = int(w/pow(2, n+1))
    g = np.zeros((i, j))
    blur = cv.GaussianBlur(t, (3, 3), 0)
    t = blur
    for u in range(i):
        for v in range(j):
            x = 2*u
            y = 2*v
            g[u, v] = t[x, y]
    t = g
    G_cell[n+1, 0] = t
    cv.imwrite('G_pyramid_'+ str(n+1)+'.jpg', g)

cv.imwrite('L_pyramid_'+ str(4)+'.jpg', G_cell[4,0])
for n in range(level):
    i = int(h/pow(2, 3-n))
    j = int(w/pow(2, 3-n))
    g = np.zeros((i, j))
    t = G_cell[4-n, 0]
    for u in range(i):
        for v in range(j):
            x = int(u/2)
            y = int(v/2)
            g[u, v] = t[x, y]
    blur = cv.GaussianBlur(g, (3, 3), 0)
    if(n <= 3):
        l = G_cell[3-n, 0] - blur
        cv.imwrite('L_pyramid_'+ str(3-n)+'.jpg', l)