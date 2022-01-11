import os

main_src = './INPUT_IMAGES/'
main_image_src = './INPUT_IMAGES/Normal/'
overexposed_image_src = './INPUT_IMAGES/OverExposed/'
underexposed_image_src = './INPUT_IMAGES/UnderExposed/'
val_main_image_src = './INPUT_IMAGES/validation/Normal/'
val_overexposed_image_src = './INPUT_IMAGES/validation/OverExposed/'
val_underexposed_image_src = './INPUT_IMAGES/validation/UnderExposed/'

arr = os.listdir(main_src)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir(main_image_src)
ensure_dir(overexposed_image_src)
ensure_dir(underexposed_image_src)
ensure_dir(val_main_image_src)
ensure_dir(val_overexposed_image_src)
ensure_dir(val_underexposed_image_src)


normal_count = 0
over_count = 0
under_count = 0

for file in arr:

    if file.endswith('0.JPG'):
        if(normal_count <= 3000):
            os.rename(main_src+file, main_image_src+file)
        else:
            os.rename(main_src+file, val_main_image_src+file)
        normal_count += 1
    if file.endswith('N1.JPG') or file.endswith('N1.5.JPG'):
        if(under_count <= 6000):
            os.rename(main_src+file, underexposed_image_src+file)
        else:
            os.rename(main_src+file, val_underexposed_image_src+file)
        under_count += 1

    if file.endswith('P1.JPG') or file.endswith('P1.5.JPG'):
        if(under_count <= 6000):
            os.rename(main_src+file, overexposed_image_src+file)
        else:
            os.rename(main_src+file, val_overexposed_image_src+file)
        over_count += 1

print(over_count)
