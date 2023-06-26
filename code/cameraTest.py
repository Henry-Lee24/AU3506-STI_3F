import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
fps = cap2.get(cv2.CAP_PROP_FPS)
width = 640
height = 480
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
outVideo = cv2.VideoWriter('test_white.avi', fourcc, fps, (width, height))

while True:
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    cv2.imshow("image1",frame1)
    cv2.imshow("image2",frame2)
    outVideo.write(frame2)
    cv2.waitKey(1)

