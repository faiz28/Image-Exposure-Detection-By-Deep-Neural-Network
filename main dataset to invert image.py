from PIL import Image
import PIL.ImageOps    
import os


DIR = './Main Database/'
normal  = DIR+'Normal/'
overexposed  = DIR+'OverExposed/'
underexposed  = DIR+'UnderExposed/'
normal_dir  = './invert-image/Normal/'
overexposed_dir  = './invert-image/OverExposed/'
underexposed_dir  = './invert-image/UnderExposed/'






def TransferFrequencyDomain(dir,destination_dir):
    imgList = os.listdir(dir)
    failed_img = 0
    for i in range(len(imgList)):
        try:
            image = Image.open(dir+imgList[i])
            inverted_image = PIL.ImageOps.invert(image)
            inverted_image.save(destination_dir+'%s'%imgList[i])
        except Exception:
            failed_img+=1
            print("%s =====> not work properly\n"%imgList[i])
    print("%s failed total  = %s"%(dir,failed_img))



TransferFrequencyDomain(normal,normal_dir)
TransferFrequencyDomain(overexposed,overexposed_dir)
TransferFrequencyDomain(underexposed,underexposed_dir)

