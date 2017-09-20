#导言区
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
import matlab.engine
import traceback


#标准函数定义区
def grayscale(img,IsShow = False):
    """图像灰度化处理函数。输入一个图片
    Or use BGR2GRAY if you read an image with cv2.imread()
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if IsShow:
        ShowImage(gray,True)
    return gray
    
def ShowImage(img,IsGray = False):
    #暂停程序并显示图片
    if IsGray:
        Fig = plt.imshow(img, cmap='gray')
    else:
        Fig = plt.imshow(img)
    plt.show()
    return Fig

def canny(img, low_threshold = 100, high_threshold = 200, IsShow = False):
    """执行灰度梯度变换"""
    edges = cv2.Canny(img,low_threshold,high_threshold)
    if IsShow:
        ShowImage(edges,True)
    return edges

def region_of_interest(img, vertices):
    """
    执行图像掩模处理，仅保留感兴趣的部分。
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def Left_Right_eye(img):
    """
    将图像分为两个部分，相当于左右眼
    并分别执行Mask
    """
    L_eye = np.copy(img)
    R_eye = np.copy(img)
    imshape = img.shape
    #掩模的中心偏移量
    Offset = 10
    #掩模的水平偏移量
    Level = 415
    #掩模的底部偏移量
    Bottom = 0.85
    #设定顶点
    vertices_L = np.array([[(200,imshape[0]*Bottom),((imshape[1]/2 - Offset),Level),((imshape[1]/2),Level),((imshape[1]/2),imshape[0]*Bottom)]],dtype = np.int)
    vertices_R = np.array([[((imshape[1]/2),imshape[0]*Bottom),((imshape[1]/2),Level),((imshape[1]/2 + Offset),Level),(imshape[1],imshape[0]*Bottom)]],dtype = np.int)

    L_eye = region_of_interest(L_eye,vertices_L)
    R_eye = region_of_interest(R_eye,vertices_R)
    return L_eye,R_eye

def hough_lines(img, rho = 1, theta = np.pi/180, threshold = 1, min_line_len = 18, max_line_gap = 10):
    """
    `img` should be the output of a Canny transform.
    利用hough算法计算图中的直线
    #距离和角度的解析度
    rho
    theta
    多少个焦点认为是一条线
    threshold
    最小线段的长度
    min_line_len
    最大线间距，超过这个间距则认为是一条线
    max_line_gap
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    return lines

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    给出直线的两点式，在图上绘制这些直线
    lines的每一行均是x1,y1,x2,y2
    """
    #生成一个全黑图像，用来绘制检测到的线段
    line_image = np.copy(img)*0
    for line in lines:
        [x1,y1,x2,y2] = line
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(img,0.8,line_image,1,0)

def CompressHoughLines(lines,MatrixIn = []):
    #将Hough算法产生的线条压缩为正常矩阵的形式
    AMatrix = MatrixIn
    for line in lines:
        for x1,y1,x2,y2 in line:
            AMatrix.append([x1,y1,x2,y2])
    return AMatrix
#主执行逻辑
#读入测试文件
#文件路径
#A_Figure = 'test_images/solidWhiteRight.jpg'
#A_Figure = 'test_images/solidWhiteCurve.jpg'
#A_Figure = 'test_images/solidYellowCurve.jpg'
#A_Figure = 'test_images/solidYellowCurve2.jpg'
#A_Figure = 'test_images/solidYellowLeft.jpg'
#A_Figure = 'test_images/whiteCarLaneSwitch.jpg'
#A_Movie = 'test_videos/solidYellowLeft.mp4'
#A_Movie = 'test_videos/solidWhiteRight.mp4'
A_Movie = 'test_videos/challenge.mp4'


CAP = cv2.VideoCapture(A_Movie)
[ret, frame] = CAP.read()
eng = matlab.engine.connect_matlab()
MatlabFig = eng.figure()
while ret:
    [ret, frame] = CAP.read()
    TempFrame = np.copy(frame)
    TempFrame[:,:,0] = np.copy(frame[:,:,2])
    TempFrame[:,:,2] = np.copy(frame[:,:,0])
    frame = np.copy(TempFrame)
    frame = grayscale(frame,False)
    Original = np.copy(frame)
    frame = canny(frame,IsShow = False)
    [L_eye,R_eye] = Left_Right_eye(frame)
    L_eye_Lines = hough_lines(L_eye)
    R_eye_Lines = hough_lines(R_eye)
    L_eye_Lines = CompressHoughLines(lines = L_eye_Lines,MatrixIn = [])
    R_eye_Lines = CompressHoughLines(lines = R_eye_Lines,MatrixIn = [])
    L_eye_Lines = matlab.int32(L_eye_Lines)
    R_eye_Lines = matlab.int32(R_eye_Lines)
    L_eye_Lines = eng.MatlabProcess(L_eye_Lines)
    R_eye_Lines = eng.MatlabProcess(R_eye_Lines)
    Lines = eng.vertcat(L_eye_Lines,R_eye_Lines)
    Original = draw_lines(Original,Lines)

    plt.imshow(Original)
    plt.ion()
    #plt.ioff()
    plt.show()
    plt.pause(0.1)
    #清除Matlab的显示
    eng.clf(MatlabFig)


#image = mping.imread(A_Figure)
#print('这是原始图像。')
#ShowImage(image)
#print('这是灰度化以后的图像。')
#image = grayscale(image,False)
#Original = np.copy(image)
#print('这是梯度检测以后的图像。')
#image = canny(image,IsShow = False)
#[L_eye,R_eye] = Left_Right_eye(image)
#print('下面是掩模以后的左右眼图像。')
#ShowImage(L_eye)
#ShowImage(R_eye)
#print('下面是利用Hough方法找到线')
#L_eye_Lines = hough_lines(L_eye)
#R_eye_Lines = hough_lines(R_eye)
#
#L_eye_Lines = CompressHoughLines(lines = L_eye_Lines,MatrixIn = [])
#R_eye_Lines = CompressHoughLines(lines = R_eye_Lines,MatrixIn = [])

#下面转到Matlab里面处理，matlab的调试系统做得很好，适应于算法开发
#eng = matlab.engine.start_matlab("-desktop")
#eng = matlab.engine.connect_matlab()
#L_eye_Lines = matlab.int32(L_eye_Lines)
#R_eye_Lines = matlab.int32(R_eye_Lines)
#L_eye_Lines = eng.MatlabProcess(L_eye_Lines)
#R_eye_Lines = eng.MatlabProcess(R_eye_Lines)
#Lines = eng.vertcat(L_eye_Lines,R_eye_Lines)
#Original = draw_lines(Original,Lines)
#ShowImage(Original)
#Temp = 0;



