import cv2
import numpy as np


def detect_left_and_right(frame):
    # 读入待匹配的图像和多个模板图像
    img = frame
    #cv2.imshow('pic',img)
    templates1 = []
    templates2 = []
    templates3 = []
    max_value = []
    # 定义一个字典或者列表，保存每个模板图像对应的名称或索引
    template_names = {1: 'left', 2: 'right', 3: 'white'}
    

    #左模板
    for i in range(1, 31):
        template = cv2.imread(f'./Left/template_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        templates1.append(template)    
    # 对每个模板图像进行模板匹配
    results1 = []
    for template in templates1:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results1.append(max_val)
    max_v = max(results1)
    max_value.append(max_v)
    
    #右模板
    for i in range(1, 32):
        template = cv2.imread(f'./Right/template_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        templates2.append(template)    
    # 对每个模板图像进行模板匹配
    results2 = []
    for template in templates2:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results2.append(max_val)
    max_v = max(results2)
    max_value.append(max_v)

    
    #白模板
    for i in range(1,11):
        template = cv2.imread(f'./White/template_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        templates3.append(template)    
    # 对每个模板图像进行模板匹配
    results3 = []
    for template in templates3:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results3.append(max_val)
    max_v = max(results3)
    max_value.append(max_v)
    print(max_value)
    
    
    # 遍历保存模板图像名称或索引的字典或列表，找到与最佳匹配结果对应的模板图像
    best_match_index = np.argmax(max_value)
    best_match_name = template_names[best_match_index+1]
   
    # 返回最佳匹配结果对应的模板图像的名称或索引
    print(f'Best match template: {best_match_name}')



cap = cv2.VideoCapture('left.avi')
while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_left_and_right(img_gray)