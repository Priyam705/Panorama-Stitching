#!/usr/bin/env python
# coding: utf-8






import cv2
import numpy as np
import argparse
import harris_17
import matplotlib.pyplot as plt
import imutils
import os
import time
cv2.ocl.setUseOpenCL(False)




def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None




def GetHomographyMatrix(LeftImage,RightImage):
    #Here we have to warp the RightImage hence homography matrix will be calculated according to that
    sift=cv2.SIFT_create()
    gray1=cv2.cvtColor(LeftImage, cv2.COLOR_RGB2GRAY)
    gray2=cv2.cvtColor(RightImage, cv2.COLOR_RGB2GRAY)
    
    corners1=harris_17.harris_corner_detector(LeftImage)
    corners2=harris_17.harris_corner_detector(RightImage)
    
    keypoints1=[cv2.KeyPoint(x[1],x[0],1) for x in corners1]
    keypoints2=[cv2.KeyPoint(x[1],x[0],1) for x in corners2]
    
    kp1,des1 = sift.compute(gray1,keypoints1)
    kp2,des2 = sift.compute(gray2,keypoints2)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) 
    best_matches = bf.match(des2,des1)
    
    M = getHomography(kp2, kp1, des2, des1, best_matches, reprojThresh=4)
    if M is None:
        print("Error!")
        assert 1==1, "No matches found";
    (matches, H, status) = M
    return H





def Blend2Images(LeftImage,RightImage):
    (r1,c1,ch1)=LeftImage.shape;
    (r2,c2,ch2)=RightImage.shape;
    assert r1==r2 and c1==c2, "Image dimensions do not match while blending two images";
    for x in range(c1):
        for y in range(r1):
            if(LeftImage[y][x][0]==0 and LeftImage[y][x][1]==0 and LeftImage[y][x][2]==0):
                LeftImage[y][x]=RightImage[y][x]
    return LeftImage




def ReadImages(path):
    listname=os.listdir(path)
    print("Reading",len(listname), "Images from",path,"...\n")
    Images=[];
    for name in listname:
        img=cv2.imread(path+'/'+name);
        Images.append(img)
    return Images




def PreProcessingImages(ImageList):
    print("Preprocessing all the Images . . .")
    (r,c,ch)=ImageList[0].shape;
    #All images size should be equal to first image and if not equal resize them
    TotalImages=len(ImageList)
    
    for i in range(TotalImages):
        (r1,c1,ch1)=ImageList[i].shape;
        assert ch==ch1, "All the images should be 3 channel images"
        if r1!=r or c1!=c:
            assert r1==r and c1==c, "Please provide all images of same size to generate panaroma"
            #print("Resizing Image no.",i+1,"...")
            #ImageList[i]=cv2.resize(ImageList[i],(r,c))
    print("PreProcessing Done !\n")
    return




def PutImagesOnCanvas(ImageList,h,w):
    (r,c)=ImageList[0].shape[:2]
    Hshift=int((w-c)/2);
    Vshift=int((h-r)/2);
    
    T=np.array([[1,0,Hshift],[0,1,Vshift],[0,0,1]],dtype=np.float32)
    for i in range(len(ImageList)):
        ImageList[i]=cv2.warpPerspective(ImageList[i],T,(w,h));
  




def GenerateCroppedImage(result):
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    
    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)
    #crop the image to the bbox coordinates
    result = result[y:y + h-1, x:x + w-1]
    return result



def GetHomographyList(ImageList):
    #print("Generating Homography matrices . . .")
    HomoList=[]
    H=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64);
    L=len(ImageList)
    HomoList.append(H);
    for i in range(1,L):
        temp=GetHomographyMatrix(ImageList[i-1],ImageList[i])
        H=np.matmul(HomoList[i-1],temp,dtype=np.float64)
        HomoList.append(H)
    
    center=int(L/2);
    Hinv=np.linalg.inv(HomoList[center])
    for i in range(0,L):
        if(i<=center):
            HomoList[i]=np.matmul(HomoList[i],Hinv,dtype=np.float64)
        else:
            HomoList[i]=np.matmul(Hinv,HomoList[i],dtype=np.float64)
    #print("Homography matrices generated !\n")
    return HomoList





def GetWarpedImages(ImageList, HomoList,PanH,PanW):
    print("Generating warped Images . . .")
    for i in range(len(ImageList)):
        ImageList[i]=cv2.warpPerspective(ImageList[i],HomoList[i],(PanW,PanH))
    print("Warped images generated !\n")
    return ImageList




def BlendAllImages(ImageList):
    print("Stitching all Images . . .")
    result=Blend2Images(ImageList[0],ImageList[1]);
    L=len(ImageList)
    mid=int(L/2)
    result=Blend2Images(ImageList[mid],ImageList[mid+1]);
    for i in range(mid+2,L):
        print("Stitching . . .")
        result=Blend2Images(result,ImageList[i]);
    for i in range(mid-1,-1,-1):
        print("Stitching . . .")
        result=Blend2Images(result,ImageList[i]);
    #print("Final Output is ready and stored in output.png!\n")
    return result




def GetPanShape(HomoList,h,w):
    cors=[];
    cors.append(np.array([[0],[0],[1]],dtype=np.float64));
    cors.append(np.array([[w],[0],[1]],dtype=np.float64));
    cors.append(np.array([[0],[h],[1]],dtype=np.float64));
    cors.append(np.array([[w],[h],[1]],dtype=np.float64));
    Xcor=[];
    Ycor=[];
    for homo in HomoList:
        for cor in cors:
            cor=np.matmul(homo,cor,dtype=np.float64);
            cor=cor/cor[2][0]; Xcor.append(cor[0][0]); Ycor.append(cor[1][0]);
            #print(cor,'\n')
            

    xmin=min(Xcor);
    xmax=max(Xcor);
    ymin=min(Ycor);
    ymax=max(Ycor);
    PanWidth=int(xmax-xmin)
    PanHeight=int(ymax-ymin)
    return int(1.1*PanHeight), int(1.1*PanWidth)




def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
	help="Provide the path of directory where images are stored")
    args = vars(ap.parse_args())

    ImageList=ReadImages(args["path"])
    PreProcessingImages(ImageList)

    print('Calculating the dimension of Canvas . . .')
    HomoList=GetHomographyList(ImageList)
    PanHeight,PanWidth=GetPanShape(HomoList,ImageList[0].shape[0],ImageList[0].shape[1])
    print('Canvas dimension calculated !\n')

    PutImagesOnCanvas(ImageList,PanHeight,PanWidth)
    print("Generating Homography matrices . . .")
    HomoList=GetHomographyList(ImageList)
    print("Homography matrices generated !\n")
    ImageList=GetWarpedImages(ImageList,HomoList,PanHeight,PanWidth)

    result=BlendAllImages(ImageList)
    result=GenerateCroppedImage(result)
    #plt.figure(figsize=(20,10))
    #plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    cv2.imwrite(args["path"]+'_output.png',result)
    print("Final Output is ready and stored in",args["path"]+"_output.png!\n")

#call to main function
if __name__=='__main__':
	main()
