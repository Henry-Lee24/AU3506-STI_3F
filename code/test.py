import cv2
import numpy as np
# 读取图像
image = cv2.imread('./red1.png')

# 将图像从BGR格式转换为HSV格式
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print(hsv)
# 定义蓝色的HSV范围
lower_blue = np.array([110, 150, 50])
upper_blue = np.array([130, 255, 255])
# 定义红色范围（在HSV颜色空间中）
lower_red = np.array([0, 150, 160])
upper_red = np.array([25, 220, 230])
upper_red2 = np.array([160, 100, 100])
lower_red2 = np.array([190, 255, 255])

# 根据HSV范围构建蓝色掩码
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# sum_of_blue = cv2.countNonZero(mask)
# 对原始图像和掩码进行位运算

# 根据红色范围创建一个掩膜（mask）
mask = cv2.inRange(hsv, lower_red, upper_red)
# mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
# mask = cv2.bitwise_or(mask, mask2)  # 合并两个掩膜

res = cv2.bitwise_and(image, image, mask=mask)
#print(res)
# 打开文件，以写入模式 ('w') 创建文件对象
# file = open('output.txt', 'w')
# # 将内容写入文件
# file.write(res)
# # 关闭文件
# file.close()

# 显示结果
cv2.imshow('Original Image', image)
#cv2.imshow('Mask', mask)
cv2.imshow('Result', res)
#print(sum_of_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()
