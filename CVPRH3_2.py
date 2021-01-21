import numpy as np
import cv2
import glob


# # 相机标定
# criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
# # 棋盘格模板规格
# w = 11  # 11
# h = 8  # 8
# # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
# objp = np.zeros((w*h, 3), np.float32)
# objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# # 储存棋盘格角点的世界坐标和图像坐标对
# objpoints = []  # 在世界坐标系中的三维点
# imgpoints = []  # 在图像平面的二维点
#
# images = glob.glob('4.5cm/*.JPG')
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 找到棋盘格角点
#     ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
#     # 如果找到足够点对，将其存储起来
#     if ret:
#         cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
#         objpoints.append(objp)
#         imgpoints.append(corners)
#         # # 将角点在图像上显示
#         # cv2.drawChessboardCorners(img, (w, h), corners, ret)
#         print("Finished {}".format(fname))
#         # cv2.imshow('findCorners', img)
#         # cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 标定
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print("ret:", ret)
# print("mtx:\n", mtx)  # 内参数矩阵
# print("dist:\n", dist)  # 畸变系数
# print("rvecs:\n", rvecs)  # 旋转向量 # 外参数
# print("tvecs:\n", tvecs)  # 平移向量 # 外参数
# print("-----------------------------------------------------")

# 稀疏光流估计(LK)
cap = cv2.VideoCapture(r"C:\Users\unkno\Documents\Master\Course\CVPR\homework\data_H3\GOPR0110.MP4")

feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=7)
lk_params = dict(winSize=(15, 15), maxLevel=2)
color = np.random.randint(0, 255, (20, 3))

# 读取第一帧图像
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

out = cv2.VideoWriter('output.avi', 0, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (old_gray.shape[1]
                                                                                          , old_gray.shape[0]))
# 返回所有检测特征点，需要输入图像，
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个mask
mask = np.zeros_like(old_frame)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        continue
    # LK光流估计-需要传入前一帧和当前图像以及前一帧检测到的角点
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1表示仍存在于两帧图像中的特征点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    out.write(img)
    if frame_idx == 100:
        cv2.imwrite("100.jpg", img)
    if frame_idx == 500:
        cv2.imwrite("500.jpg", img)
    # cv2.imshow('frame', img)
    # k = cv2.waitKey(150) & 0xff
    # if k == 27:
    #     break

    # 相机运动估计（对相邻每两帧进行估计）
    # 1. 进行相机标定获取内参数
    K = np.array([[1.75336237e+03, 0.00000000e+00, 2.04376746e+03], [0.00000000e+00, 1.75278349e+03, 1.54908719e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])  # 相机内参矩阵
    # 2. 根据两帧间的配对点对估计本质矩阵
    essential_matrix, _ = cv2.findEssentialMat(good_new, good_old, cameraMatrix=K)
    # 3. 进行矩阵分解完成运动估计
    if essential_matrix is not None:
        ret, R, t, _ = cv2.recoverPose(essential_matrix, good_new, good_old, cameraMatrix=K)
        print("Rotation: {0} at frame {1}".format(R, frame_idx))
        print("Translation: {0} at frame {1}".format(t, frame_idx))

    if len(good_new) < 8:
        # 增加特征点
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    else:
        # 更新特征点
        p0 = good_new.reshape(-1, 1, 2)
    # 更新前一帧
    old_gray = frame_gray.copy()
    frame_idx += 1


cv2.destroyAllWindows()
cap.release()
out.release()
