import cv2
import numpy as np
import math
'''
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    cv2.imshow("image1",frame1)
    cv2.imshow("image2",frame2)
    cv2.waitKey(3)
    roi = frame1

    # 灰度图
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 大津法
    retval, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('binary.jpg', binary_img)
'''


# 去反光
def illum(img):

    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_bw, 180, 255, 0)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    # img[thresh == 255] = 150
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_zero[y:y + h, x:x + w] = 255
    mask = img_zero
    # cv2.imshow("mask", mask)
    result = cv2.illuminationChange(img, mask, alpha=2, beta=2)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    return result


def frame_process(frame1):

    roi = frame1
    size = roi.shape
    height = size[0]
    width = size[1]
    roi = frame1[(int)(height / 2):(int)(height),
                 (int)(width / 5):(int)(width * 4 / 5)]

    # 开运算
    kernel = np.ones((7, 7), np.uint8)
    opening_img = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    # 去反光
    illum_img = illum(opening_img)
    # 灰度图
    gray_img = cv2.cvtColor(illum_img, cv2.COLOR_BGR2GRAY)
    # 大津法
    retval, binary_img = cv2.threshold(gray_img, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 高斯模糊
    blur_img = cv2.GaussianBlur(binary_img, (5, 5), 3)
    # 取边线
    edges = cv2.Canny(blur_img, 100, 200)
    return edges


def test_video(filename):

    cam = cv2.VideoCapture(filename)
    # 确定视频高 宽 帧率 视频总帧数
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cam.get(cv2.CAP_PROP_FPS))
    frameCount = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    # 第二步:循环得到视频帧，并写入新视频
    success, frame1 = cam.read()
    frame_count = 0  #当前写入

    # 读取视频帧
    while success:

        edges = frame_process(frame1)
        # 滑窗法
        window_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        histogram = np.sum(edges[:, :], axis=0) // 255
        # print(histogram)
        num = 0

        lane_base = np.argmax(histogram)
        # print(lane_base)
        nwindows = 20  # 滑窗个数
        window_height = int(edges.shape[0] / nwindows)  # 滑窗高度
        nonzero = edges.nonzero()  # 非0的坐标值
        # print(nonzero)
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_current = lane_base
        # print(nonzerox)
        margin = 25
        minpix = 5

        lane_inds = []

        for window in range(nwindows):
            win_y_low = edges.shape[0] - (window + 1) * window_height
            win_y_high = edges.shape[0] - window * window_height
            win_x_low = lane_current - margin
            win_x_high = lane_current + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                         & (nonzerox >= win_x_low) &
                         (nonzerox < win_x_high)).nonzero()[0]
            cv2.rectangle(window_img, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), (0, 0, 255), 3)

            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                lane_current = int(np.mean(nonzerox[good_inds]))

        # print(lane_inds)
        cv2.imshow("frame", window_img)
        # 判断用户是否有按键输入,如果有则跳出循环
        # cv2.waitKey如果有用户输入,返回输入的字符，否则返回-1
        char = cv2.waitKey(1)
        if char != -1:
            break
        # 读取新视频
        success, frame1 = cam.read()
        # frame_count = frame_count + 1

    # 释放视频读取对象
    cam.release()


def test_image():

    frame = cv2.imread("1.png")
    roi = frame
    size = roi.shape
    width = size[0]
    height = size[1]
    roi = frame[0:int(size[0]), int(size[1] / 2):int(size[1])]
    left_pt = width / 2
    right_pt = width / 2
    mid_pt = width / 2
    # 灰度图
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 大津法
    retval, binary_img = cv2.threshold(gray_img, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("frame", binary_img)
    cv2.waitKey(20000)




def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = np.sum(binary_warped[:, :], axis=0)
    lane_base = np.argmax(histogram)
    img1 = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
    num = 0
    for num in range(len(histogram) - 1):
        cv2.line(img1, (num, int(720 - histogram[num] / 200)), (num + 1, int(720 - histogram[num + 1] / 200)),
                 (255, 255, 0), 5)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 25
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            cv2.rectangle(img1, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 255), 3)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            cv2.rectangle(img1, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 255), 3)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]



    num = 0
    for num in range(len(ploty) - 1):
        cv2.line(img1, (int(left_fitx[num]), int(ploty[num])), (int(left_fitx[num + 1]), int(ploty[num + 1])),
                 (0, 0, 255), 20)
        cv2.line(img1, (int(right_fitx[num]), int(ploty[num])), (int(right_fitx[num + 1]), int(ploty[num + 1])),
                 (0, 0, 255), 20)


    vertices = np.array([[(int(left_fitx[0]), int(ploty[0])), (int(left_fitx[num - 1]), int(ploty[num - 1])),
                          (int(right_fitx[num - 1]), int(ploty[num - 1])), (int(right_fitx[0]), int(ploty[0]))]])
    cv2.fillPoly(img1, vertices, (0, 255, 0))

    aP = [0.0, 0.0]

    if (lane_base >= 640):
        LorR = -1.0  # Right
        plotx = right_fitx
        fit = right_fit

    else:
        LorR = 1.0  # Left
        plotx = left_fitx
        fit = left_fit

    aimLaneP = [int(plotx[len(ploty) // 2]), int(ploty[len(ploty) // 2])]
    img1 = cv2.circle(img1, (aimLaneP[0], aimLaneP[1]), 25, (255, 0, 0), -1)

    lanePk = (1 / (2 * fit[0])) * aimLaneP[0] - fit[1] / (2 * fit[0])
    k_ver = -1 / lanePk
    theta = math.atan(k_ver)
    aP[0] = aimLaneP[0] + math.cos(theta) * (LorR) * 665 / 2
    aP[1] = aimLaneP[1] + math.sin(theta) * (LorR) * 665 / 2

    img1 = cv2.circle(img1, (int(aP[0]), int(aP[1])), 25, (0, 0, 255), -1)
    return img1




if __name__ == '__main__':
    filename = "save_test_video.avidd.avi"
    test_video(filename)
