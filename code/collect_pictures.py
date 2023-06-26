import cv2
 
cap = cv2.VideoCapture("./test_white.avi")
c = 1
frameRate = 20  # 帧数截取间隔（每隔100帧截取一帧）
i =1
while(True):
	ret, frame = cap.read()
	
	#print(frame.shape)
	if ret:
		frame =frame[100:320,250:420]
		if(c % frameRate == 0 ):
			print("开始截取视频第：" + str(c) + " 帧")
			# 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地			
			cv2.imwrite("./White/" + 'template_'+str(i) + '.jpg', frame)  # 这里是将截取的图像保存在本地
			i +=1
		c += 1
		cv2.waitKey(0)
	else:
		print("所有帧都已经保存完成")
		break
cap.release()