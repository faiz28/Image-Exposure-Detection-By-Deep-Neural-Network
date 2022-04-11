from PIL import Image, ImageEnhance
import os

DIR = './Main Database/'

overexposed  = DIR+'OverExposed/'
underexposed  = DIR+'UnderExposed/'
overexposed_dir  = './increased and decrased exposure/OverExposed/'
underexposed_dir  = './increased and decrased exposure/UnderExposed/'

#read the image



def ChangeExposure(dir,destination_dir,factor):
    imgList = os.listdir(dir)
    failed_img = 0
    for i in range(len(imgList)):
        try:
            im = Image.open(dir+imgList[i])
            #image brightness enhancer
            enhancer = ImageEnhance.Brightness(im)
            im_output = enhancer.enhance(factor)
            im_output.save(destination_dir+'%s'%imgList[i])

        except Exception:
            failed_img+=1
            print("%s =====> not work properly\n"%imgList[i])
    print("%s failed total  = %s"%(dir,failed_img))
     

ChangeExposure(overexposed,overexposed_dir,factor = 1.3)
ChangeExposure(underexposed,underexposed_dir,factor = 0.7)

