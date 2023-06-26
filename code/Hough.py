import cv2
import numpy as np
num = 0
i = 20
# color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
#               'blue': {'Lower': np.array([80, 100, 120]), 'Upper': np.array([130, 220, 180])},
#               'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
#               'yellow': {'Lower': np.array([28, 160, 115]), 'Upper': np.array([32, 250, 255])},
#               'yellow2': {'Lower': np.array([25, 100, 230]), 'Upper': np.array([45, 180, 255])}}

def hough_detect(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # OTSU
    retval, binary_img = cv2.threshold(gray_img, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    gray = gray_img[100:350,200:420] # 480*640
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 0.01, param1=100, param2=200, minRadius=1, maxRadius=130)
    obstacle_detected = 0
    if circles is not None:
        circles = circles[0, :].astype(int)
        x_back = None
        y_back = None
        for x, y, r in circles:
            x = x + 200
            y = y + 100
            x_back = x
            y_back = y
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            roi = frame[(y-r):(y+r),
                  (x - r):(x + r) ]
            hsv_img2 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            #print("circle!")
    cv2.imshow("binary_img",binary_img)
    cv2.waitKey(1)
    return x_back,y_back

    #if cv2.waitKey(100) == ord('q'):
        #break


def detect_left_and_right(frame):

    img = frame
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # OTSU
    retval, img = cv2.threshold(img, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #k = np.ones((10, 10), np.uint8)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN,k)
    # cv2.imshow('pic',img)
    templates1 = []
    templates2 = []
    templates3 = []
    max_value = []

    template_names = {1: 'left', 2: 'right', 3: 'white'}


    for i in range(0, 33):
        template = cv2.imread('./Left/template_'+str(i)+'.jpg', cv2.IMREAD_GRAYSCALE)
        #gray_img = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # OTSU
        retval, template = cv2.threshold(template, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #k = np.ones((10, 10), np.uint8)
        #template = cv2.morphologyEx(template, cv2.MORPH_OPEN,k)
        cv2.imshow("tempalte",template)
        templates1.append(template)

    results1 = []
    for template in templates1:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results1.append(max_val)
    max_v = max(results1)
    max_value.append(max_v)


    for i in range(0, 21):
        template = cv2.imread('./Right/template_'+str(i)+'.jpg', cv2.IMREAD_GRAYSCALE)
        #gray_img = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # OTSU
        retval, template = cv2.threshold(template, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #k = np.ones((10, 10), np.uint8)
        #template = cv2.morphologyEx(template, cv2.MORPH_OPEN,k)
        templates2.append(template)

    results2 = []
    for template in templates2:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results2.append(max_val)
    max_v = max(results2)
    max_value.append(max_v)


    for i in range(1, 8):
        template = cv2.imread('./White/template_'+str(i)+'.jpg', cv2.IMREAD_GRAYSCALE)
        templates3.append(template)

    results3 = []
    for template in templates3:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results3.append(max_val)
    max_v = max(results3)
    max_value.append(max_v)
    best_match_index = np.argmax(max_value)
    max_max_value = max(max_value)
    #print(max_max_value)
    best_match_name = template_names[best_match_index + 1]    
    print('Best match template: ', best_match_name)
    print('max_max_value',max_max_value)
    if max_max_value > 0.15 and best_match_index == 0:
        return best_match_index+1
    elif max_max_value > 0.3 and best_match_index == 1:
        return best_match_index+1
    else :
        return 0