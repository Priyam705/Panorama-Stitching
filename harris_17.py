import cv2
import numpy as np
def harris_corner_detector(input_img, NMS=True, window='gaussian'):  
    """
    -This function takes RBG image as input and returns the list of (x,y) co-ordinates of corner points detected by harris corner
     detection algorithm.
    -valid values for NMS are {True, False}
    -valid values for window are {gaussian, unit}
    """
    # check every variable has valid inputs
    assert window in {'unit','gaussian'}, "Value of window should be from {unit, gaussian}";
    assert NMS in {True, False}, "Value of NMS should be from {True, False}" ;
    
    #Converting the image into gray scale because harris corner detection works on gray scale images
    input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY);

    (r,c)=input_img.shape[:2]
    
    #Finding gradient in x and y direction which are intermediate variables for calculating structure tensor 'M'
    Ix=cv2.Sobel(input_img,cv2.CV_64F,1,0,ksize=3)
    Iy=cv2.Sobel(input_img,cv2.CV_64F,0,1,ksize=3)
    Ixx=Ix*Ix
    Iyy=Iy*Iy
    Ixy=Ix*Iy
    
    if window=='gaussian':
        Ixx=cv2.GaussianBlur(Ixx,(3,3),0)
        Iyy=cv2.GaussianBlur(Iyy,(3,3),0)
        Ixy=cv2.GaussianBlur(Ixy,(3,3),0)
    else:
        kernel=np.ones((3, 3))
        Ixx=cv2.filter2D(Ixx,-1,kernel);
        Iyy=cv2.filter2D(Iyy,-1,kernel);
        Ixy=cv2.filter2D(Ixy,-1,kernel);

    #R matrix stores harris corner response value for each corresponding pixel of input image
    k=0.04
    R=np.zeros((r,c),dtype=np.float64)
    R=((Ixx*Iyy)-(Ixy*Ixy))-(k*(Ixx+Iyy)*(Ixx+Iyy))
    
    thres=0.01*R.max()
    corner_points=[]

    for i in range(1,r-1):
        for j in range(1,c-1):
            if NMS:
                if R[i,j]>thres and R[i,j]>=R[i,j-1] and R[i,j]>=R[i-1,j] and R[i,j]>=R[i,j+1]  and R[i,j]>=R[i+1,j]  and R[i,j]>=R[i+1,j+1]  and R[i,j]>=R[i-1,j-1]  and R[i,j]>=R[i+1,j-1]  and R[i,j]>=R[i-1,j+1]:
                    corner_points.append([i,j])
            else:
                if R[i,j]>thres:
                    corner_points.append([i,j])
    return corner_points





