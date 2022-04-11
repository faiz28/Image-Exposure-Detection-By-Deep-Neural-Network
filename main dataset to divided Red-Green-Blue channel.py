import cv2
import os
import numpy as np

DIR = './Main Dataset/'

total_img = 2970

def prepare_image_array(imgDir, imgW, imgH):
    imgList = os.listdir(imgDir)
    # print(imgList)
    n = len(imgList)

    imgSet = []
    for i in range(total_img):
        imgPath = imgDir + imgList[i]
        if (os.path.exists(imgPath)):
            # print(imgPath)
            img = cv2.imread(imgPath)
            resizedImg = cv2.resize(img, (imgW, imgH))
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            imgSet.append(rgbImg)
        else:
            print("It is not a valid image path.")

    print("total image "+str(len(imgSet)))
    imgSet = np.array(imgSet, dtype=np.uint8)
    print("total shape "+str(imgSet.shape))

    return imgSet

imgDir = DIR+'Normal/'
imgSet1 = prepare_image_array(imgDir, 512, 512)
m = imgSet1.shape[0]
imgDir = DIR+'OverExposed/'
imgSet2 = prepare_image_array(imgDir, 512, 512)
n = imgSet2.shape[0]

imgDir = imgDir = DIR+'UnderExposed/'
imgSet3 = prepare_image_array(imgDir, 512, 512)
o = imgSet3.shape[0]

imgSet  = np.concatenate([imgSet1, imgSet2, imgSet3]);

print("toal image  ====== " + str(imgSet.shape))



#read image
# src = cv2.imread('D:/cv2-resize-image-original.png', cv2.IMREAD_UNCHANGED)
# print(src.shape)

# #extract red channel


blue_normal = os.path.join('./blue-channel/', 'Normal')
blue_UnderExposed = os.path.join('./blue-channel/', 'UnderExposed')
blue_OverExposed = os.path.join('./blue-channel/', 'OverExposed')
if(os.path.exists(blue_normal) == False):
    os.mkdir(blue_normal)
if(os.path.exists(blue_UnderExposed) == False):
    os.mkdir(blue_UnderExposed)
if(os.path.exists(blue_OverExposed) == False):
    os.mkdir(blue_OverExposed)

green_normal = os.path.join('./green-channel/', 'Normal')
green_UnderExposed = os.path.join('./green-channel/', 'UnderExposed')
green_OverExposed = os.path.join('./green-channel/', 'OverExposed')
if(os.path.exists(green_normal) == False):
    os.mkdir(green_normal)
if(os.path.exists(green_UnderExposed) == False):
    os.mkdir(green_UnderExposed)
if(os.path.exists(green_OverExposed) == False):
    os.mkdir(green_OverExposed)
    
red_normal = os.path.join('./red-channel/', 'Normal')
red_UnderExposed = os.path.join('./red-channel/', 'UnderExposed')
red_OverExposed = os.path.join('./red-channel/', 'OverExposed')
if(os.path.exists(red_normal) == False):
    os.mkdir(red_normal)
if(os.path.exists(red_UnderExposed) == False):
    os.mkdir(red_UnderExposed)
if(os.path.exists(red_OverExposed) == False):
    os.mkdir(red_OverExposed)




count=0
for  i in range(0, len(imgSet)):

    # #write red channel to greyscale image
    
    if(count<total_img):
        red_channel = imgSet[i][:,:,2]
        red_img = np.zeros(imgSet[i].shape)
        red_img[:,:,2] = red_channel
        cv2.imwrite(red_normal+'/image - %s.jpg'%count,red_img) 

        green_channel = imgSet[i][:,:,1]
        green_img = np.zeros(imgSet[i].shape)
        green_img[:,:,1] = green_channel
        cv2.imwrite(green_normal+'/image - %s.jpg'%count,green_img) 

        blue_channel = imgSet[i][:,:,0]
        blue_img = np.zeros(imgSet[i].shape)
        blue_img[:,:,0] = blue_channel
        cv2.imwrite(blue_normal+'/image - %s.jpg'%count,blue_img) 
        
    elif(count>= total_img and count<2*total_img):
        red_channel = imgSet[i][:,:,2]
        red_img = np.zeros(imgSet[i].shape)
        red_img[:,:,2] = red_channel
        cv2.imwrite(red_OverExposed+'/image - %s.jpg'%count,red_img) 

        green_channel = imgSet[i][:,:,1]
        green_img = np.zeros(imgSet[i].shape)
        green_img[:,:,1] = green_channel
        cv2.imwrite(green_OverExposed+'/image - %s.jpg'%count,green_img) 

        blue_channel = imgSet[i][:,:,0]
        blue_img = np.zeros(imgSet[i].shape)
        blue_img[:,:,0] = blue_channel
        cv2.imwrite(blue_OverExposed+'/image - %s.jpg'%count,blue_img) 
        
    else:
        red_channel = imgSet[i][:,:,2]
        red_img = np.zeros(imgSet[i].shape)
        red_img[:,:,2] = red_channel
        cv2.imwrite(red_UnderExposed+'/image - %s.jpg'%count,red_img) 

        green_channel = imgSet[i][:,:,1]
        green_img = np.zeros(imgSet[i].shape)
        green_img[:,:,1] = green_channel
        cv2.imwrite(green_UnderExposed+'/image - %s.jpg'%count,green_img) 

        blue_channel = imgSet[i][:,:,0]
        blue_img = np.zeros(imgSet[i].shape)
        blue_img[:,:,0] = blue_channel
        cv2.imwrite(blue_UnderExposed+'/image - %s.jpg'%count,blue_img)  
    count+=1
    print(count)