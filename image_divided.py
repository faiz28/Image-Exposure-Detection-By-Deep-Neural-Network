import os
DIR = './Main Dataset/'


# after download the dataset it was in "INPUT_IMAGES", then we split into three differnet folder
main_src = './INPUT_IMAGES/'
# split all data into three different dataset
main_image_src = DIR+'Normal/'
overexposed_image_src = DIR+'OverExposed/'
underexposed_image_src = DIR+'UnderExposed/'

arr = os.listdir(main_src)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir(main_image_src)
ensure_dir(overexposed_image_src)
ensure_dir(underexposed_image_src)

normal_count = 0
over_count = 0
under_count = 0

for file in arr:
    if file.endswith('0.JPG'):
        os.rename(main_src+file, main_image_src+file)
    if  file.endswith('N1.5.JPG'):
        os.rename(main_src+file, underexposed_image_src+file)
    if  file.endswith('P1.5.JPG'):
        os.rename(main_src+file, overexposed_image_src+file)
      

