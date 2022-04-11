import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
import os
import cv2
DIR = './Main Dataset/'
normal  = DIR+'Normal/'
overexposed  = DIR+'OverExposed/'
underexposed  = DIR+'UnderExposed/'
normal_dir  = './frequency/Normal/'
overexposed_dir  = './frequency/OverExposed/'
underexposed_dir  = './frequency/UnderExposed/'

count = 0

def TransferFrequencyDomain(dir,destination_dir):
    imgList = os.listdir(dir)
    failed_img = 0
    for i in range(len(imgList)):
        try:
            dark_image = imread(dir+imgList[i])
            dark_image_grey = rgb2gray(dark_image)
            dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))    
            img  = np.log(abs(dark_image_grey_fourier))
            # print("%s %s\n"%(img.shape[0]/100,img.shape[1]/100))
            plt.figure(num=None, figsize=((img.shape[1]/100),(img.shape[0]/100)), dpi=80)
            plt.imshow(img)
            plt.savefig(destination_dir+'%s'%imgList[i],dpi=100)
            # cv2.imwrite(normal_dir+'/image - %s.jpg'%imgList[i],dark_image)
            plt.imshow(dark_image, cmap='gray')
        except Exception:
            failed_img+=1
            print("%s =====> not work properly\n"%imgList[i])
    print("%s failed total  = %s"%(dir,failed_img))



# TransferFrequencyDomain(normal,normal_dir)
TransferFrequencyDomain(overexposed,overexposed_dir)
# TransferFrequencyDomain(underexposed,underexposed_dir)

