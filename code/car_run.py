import numpy as np
import cv2
import math
from driver import driver
import time
from Hough import hough_detect,detect_left_and_right

car = driver()

    
# PID
class PID_Controller:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error = 0
        self.error_sum = 0
        self.error_diff = 0

    def __call__(self, error):
        self.error_sum = self.error_sum + error
        self.error_diff = error - self.error
        self.error = error
        u = self.kp * self.error + self.ki * self.error_sum + self.kd * self.error_diff
        return u


# 
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


# 
def find_road(frame1):
    roi = frame1
    size = roi.shape
    height = size[0]
    width = size[1]
    roi = frame1[(int)(height / 2):(int)(height),
          (int)(width / 5):(int)(width * 4 / 5)]

    # open
    kernel = np.ones((7, 7), np.uint8)
    opening_img = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    # 
    #illum_img = illum(opening_img)
    # 
    gray_img = cv2.cvtColor(opening_img, cv2.COLOR_BGR2GRAY)
    # OTSU
    retval, binary_img = cv2.threshold(gray_img, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 
    # blur_img = cv2.GaussianBlur(binary_img, (5, 5), 3)
    # 
    # edges = cv2.Canny(blur_img, 100, 200)
    return binary_img


def find_midpoint(img):
    window_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height = img.shape[0]
    width = img.shape[1]
    nwindows = 20  # 
    window_width = 60
    window_height = int(height / nwindows)  # 
    step = 3
    y_far = height - 4 * window_height - 1
    y_near = height - 2 * window_height - 1
    y_flag = 0
    y = y_far
    '''
    if y_flag == 0:
        y = y_far
    else:
        y = y_near
    '''
    pixel_sum = []
    for x in range(0, width - window_width - 1, step):
        pixel_sum.append(sum(sum(img[y - window_height:y,
                                 x:x + window_width]))) 
        # print(sum(sum(img[y - window_height:y, x:x + window_width])))
        # cv2.waitKey(20)
        # cv2.rectangle(window_img, (x, y),
        #               (x + window_width, y - window_height), (0, 0, 255), 2)
        # cv2.imshow("window_img", window_img)

    max_value = max(pixel_sum)
    # print(max_value)
    if max_value > 14500:
        # y_flag = (y_flag + 1) % 2
        y_flag = 1
    else:
        y_flag = 0
    # print(y_flag)
    num = 0
    position = 0
    for i in range(len(pixel_sum)):
        if pixel_sum[i] == max_value:
            num = num + 1
            position = position + i
    # print(num)
    position = int(position / num) * step
    # position = np.argmax(pixel_sum) * step
    cv2.rectangle(window_img, (position, y - window_height),
                  (position + window_width, y), (0, 0, 255), 2)
    cv2.imshow("window_img", window_img)
    return position + window_width / 2, y - window_height / 2, y_flag


def speed_control(aim_pt_x, aim_pt_y, pos_x, pos_y):
    speed = 60
    dx = aim_pt_x - pos_x
    dy = pos_y - aim_pt_y

    angle_error = math.atan2(dx, dy)
    # print(angle_error)
    pid = PID_Controller(0.17, 0.00023, 0.16)
    diff_speed = pid(angle_error) * speed
    left_speed = max(speed + diff_speed, 0)
    right_speed = max(speed - diff_speed, 0)
    #print(left_speed, right_speed)
    return left_speed, right_speed


def frame_process(frame):
    road_img = find_road(frame)
    mid_pt_x, mid_pt_y, flag = find_midpoint(road_img)
    left_speed, right_speed = speed_control(mid_pt_x, mid_pt_y,
                                            road_img.shape[1] / 2 - 1, road_img.shape[0] - 1)
    return left_speed, right_speed, flag


def test_video(filename):
    cam = cv2.VideoCapture(filename)
    # 
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cam.get(cv2.CAP_PROP_FPS))
    frameCount = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    # 
    success, frame1 = cam.read()
    frame_count = 0  #

    #
    while success:

        frame_process(frame1)

        # cv2.imshow("frame", window_img)
        
        char = cv2.waitKey(1)
        if char != -1:
            break
        # 
        success, frame1 = cam.read()
        # frame_count = frame_count + 1

    # 
    cam.release()


if __name__ == '__main__':
    '''## 
    filename = "save_test_video.avidd.avi"
    test_video(filename)
    '''
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    list = [0]
    max_num = 10
    time_num = 70
    count = 0
    direction = 0
    flag_full = 0
    flag = 0
    while True:
        print(car.read_battery())
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        #cv2.imshow("image1",frame1)
        cv2.imshow("image2",frame2)
        #start = time.time()
        #frame_bin = find_road(frame2)
        #end = time.time()
        #print('waste time:',end-start)
        #cv2.imshow("binary_img", frame_bin)
        cv2.waitKey(1)
        #left_speed = right_speed = 10
        if flag == 0:
            left_speed, right_speed, flag = frame_process(frame2)
        #car.set_speed(0, 0)
        x = None
        y = None
        '''
        
        print(car.read_battery())
        '''
        '''
        
        '''
        
        #print("flag:",flag)
        if flag == 1:
            
            car.set_speed(0, 0)
            
            
            try:
                x,y = hough_detect(frame2)
            except:
                pass
            if x is not None:
                img_gray = cv2.cvtColor(frame2[y-50:y+50,x-50:x+50], cv2.COLOR_BGR2GRAY)
                temp = detect_left_and_right(img_gray)
                print("temp",temp)
                if temp == 1 or temp ==2:
                    if len(list) < max_num:
                        list.append(temp)
                        print("list:",list)
                    elif len(list) == max_num:
                        flag_full = 1
                        
            
            if flag_full:
                direction = np.argmax(np.bincount(list))
                print("direction:",direction)
                if direction == 1:
                    while count < time_num:
                        _, frame1 = cap1.read()
                        _, frame2 = cap2.read()
                        #cv2.imshow("image1",frame1)
                        cv2.imshow("image2",frame2)
                        car.set_speed(45, 5)
                        count += 1
                    count = 0
                    flag = 0
                    list=[0]
                    flag_full = 0
                elif direction ==2:
                    while count < time_num:
                        _, frame1 = cap1.read()
                        _, frame2 = cap2.read()
                        #cv2.imshow("image1",frame1)
                        cv2.imshow("image2",frame2)
                        car.set_speed(5, 45)
                        count += 1               
                    count = 0
                    list=[0]
                    flag = 0
                    flag_full = 0
        else:
            car.set_speed(right_speed, left_speed)
            #car.set_speed(0, 0)