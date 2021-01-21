import cv2
import math
import numpy as np

f = cv2.imread(r"C:\Users\unkno\Documents\Master\Course\ImageProcessing\C--\lena512.jpg")
def my_SIFT(f):
    h, w = f.shape[:2]
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    m_g = np.zeros((h, w))
    a_g = np.zeros((h, w))
    aa_g = np.zeros((h, w))
    nms_g = np.zeros((h, w))
    point_g = np.zeros((h, w))
    show_img = np.zeros((h, w))
    x_sd = np.zeros((8, 1))
    feature = np.zeros((128, 1))
    key_point = np.zeros((h, w), dtype=object)

    level = 6
    #######################################
    thr_m = 50
    #######################################
    G_cell = np.empty((level, 1), dtype=object)
    DoG_cell = np.empty((level, 1), dtype=object)


    G_cell[0, 0] = f
    t = G_cell[0, 0]

    for n in range(level-1):
        i = int(h/pow(2, n+1))
        j = int(w/pow(2, n+1))
        g = np.zeros((i, j))
        blur = cv2.GaussianBlur(t, (3, 3), math.pow(n, 0.5))
        t = blur
        for u in range(i):
            for v in range(j):
                x = 2*u
                y = 2*v
                g[u, v] = t[x, y]
        t = g
        G_cell[n+1, 0] = t

    for m in range(level):
        lap = cv2.Laplacian(G_cell[m, 0], cv2.CV_64F)
        for n in range(level):
            blur = cv2.GaussianBlur(lap, (9, 9), 1.6*math.pow(n, 0.5))
            t = blur
            DoG_cell[n, 0] = t

        for i in range(1, level-1):
            sd = DoG_cell[i, 0]
            sd_a = DoG_cell[i-1, 0]
            sd_b = DoG_cell[i+1, 0]
            for x in range(3, int((h/math.pow(2, m))-3)):
                for y in range(3, int((w/math.pow(2, m))-3)):
                    index = 0
                    sdd = sd[x, y]
                    x_sd_a = sd_a[x - 3:x + 2, y - 3:y + 2]
                    x_sd_b = sd_b[x - 3:x + 2, y - 3:y + 2]
                    x_sd[0, 0] = sd[x-1, y]
                    x_sd[1, 0] = sd[x, y-1]
                    x_sd[2, 0] = sd[x, y+1]
                    x_sd[3, 0] = sd[x+1, y]
                    x_sd[4, 0] = sd[x-1, y-1]
                    x_sd[5, 0] = sd[x-1, y+1]
                    x_sd[6, 0] = sd[x+1, y-1]
                    x_sd[7, 0] = sd[x+1, y+1]
                    max_1 = sdd - x_sd_a.max()
                    max_2 = sdd - x_sd_b.max()
                    max_3 = sdd - x_sd.max()
                    min_1 = sdd - x_sd_a.min()
                    min_2 = sdd - x_sd_b.min()
                    min_3 = sdd - x_sd.min()

                    if (max_1>=0 and max_2>=0 and max_3>=0):
                        u = x*math.pow(2, m)
                        v = y*math.pow(2, m)
                        point_g[int(u), int(v)] = 1
                    if (min_1<=0 and min_2<=0 and min_3<=0):
                        u = x*math.pow(2, m)
                        v = y*math.pow(2, m)
                        point_g[int(u), int(v)] = 1

    blur = cv2.GaussianBlur(f, (7, 7), 0)
    f1 = blur
    gy = cv2.Sobel(f1, cv2.CV_64F, 1, 0, ksize=3)
    gx = cv2.Sobel(f1, cv2.CV_64F, 0, 1, ksize=3)

    for x in range(h):
        for y in range(w):
            m_g[x, y] = (math.sqrt((pow(gx[x, y], 2)+pow(gy[x, y], 2))))

    for x in range(h):
        for y in range(w):
            a_g[x, y] = np.arctan2(gy[x, y], gx[x, y])*180/math.pi
            if (a_g[x, y] < 0):
                a_g[x, y] = a_g[x, y] + 180
                a_g[x, y] = int(a_g[x, y] / 45) + 4
            else:
                a_g[x, y] = int(a_g[x, y] / 45)

    for x in range(h):
        for y in range(w):
            if (point_g[x, y] == 1):
                if (m_g[x, y] < thr_m):
                    point_g[x, y] = 0

    ########
    #展示特征点
    for x in range(h):
        for y in range(w):
            if (point_g[x, y] == 1):
                    show_img[x, y] = 255
            else:
                show_img[x, y] = f[x, y]
    cv2.imwrite('show.jpg', show_img)
    ########

    def sub_feature(x, y, theta):
        a = np.zeros((8, 1))
        hi = np.zeros((8, 1))
        for i in range(x, x+4):
            for j in range(y, y+4):
                hi[int(a_g[i, j]), 0] = hi[int(a_g[i, j]), 0] + 1
        for m in range(int(theta), 8):
            a[int(m-theta), 0] = hi[m, 0]
        for n in range(int(theta)):
            a[int(8-theta+n), 0] = hi[n, 0]

        return a



    for x in range(8, h-8):
        for y in range(8, w-8):
            if (point_g[x, y] == 1):
                theta = a_g[x, y]
                a1 = sub_feature(x-7, y-7, theta)
                feature[0:8, 0] = a1[:, 0]
                a2 = sub_feature(x-7, y-3, theta)
                feature[8:16, 0] = a2[:, 0]
                a3 = sub_feature(x-7, y+1, theta)
                feature[16:24, 0] = a3[:, 0]
                a4 = sub_feature(x-7, y+5, theta)
                feature[24:32, 0] = a4[:, 0]
                a5 = sub_feature(x-3, y-7, theta)
                feature[32:40, 0] = a5[:, 0]
                a6 = sub_feature(x-3, y-3, theta)
                feature[40:48, 0] = a6[:, 0]
                a7 = sub_feature(x-3, y+1, theta)
                feature[48:56, 0] = a7[:, 0]
                a8 = sub_feature(x-3, y+5, theta)
                feature[56:64, 0] = a8[:, 0]
                a9 = sub_feature(x+1, y-7, theta)
                feature[64:72, 0] = a9[:, 0]
                a10 = sub_feature(x+1, y-3, theta)
                feature[72:80, 0] = a10[:, 0]
                a11 = sub_feature(x+1, y+1, theta)
                feature[80:88, 0] = a11[:, 0]
                a12 = sub_feature(x+1, y+5, theta)
                feature[88:96, 0] = a12[:, 0]
                a13 = sub_feature(x+5, y-7, theta)
                feature[96:104, 0] = a13[:, 0]
                a14 = sub_feature(x+5, y-3, theta)
                feature[104:112, 0] = a14[:, 0]
                a15 = sub_feature(x+5, y+1, theta)
                feature[112:120, 0] = a15[:, 0]
                a16 = sub_feature(x+5, y+5, theta)
                feature[120:128, 0] = a16[:, 0]
                key_point[x, y] = feature
    return key_point

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size=800

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        kp_img1 = cv2.drawKeypoints(img1,kp1,None)
        kp_img2 = cv2.drawKeypoints(img2, kp2, None)
        cv2.imwrite('test1.jpg',kp_img1)
        cv2.imwrite('test2.jpg', kp_img2)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        result=panorama1+panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

img1 = cv2.imread(r"C:\Users\unkno\Documents\Master\Course\CVPR\homework\data_H2\GoPro\outside\GOPR0101.JPG")
img2 = cv2.imread(r"C:/Users/unkno/Documents/Master/Course/CVPR/homework/data_H2/GoPro/outside/GOPR0104.JPG")
final=Image_Stitching().blending(img1, img2)
cv2.imwrite('panorama.jpg', final)

# cv.imwrite('gx.jpg', gx)
# cv.imwrite('gy.jpg', gy)
# cv.imwrite('nms.jpg', nms_g)
