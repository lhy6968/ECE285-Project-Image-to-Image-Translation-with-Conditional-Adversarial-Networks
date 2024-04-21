from PIL import Image
import numpy as np
import math

def psnr(target_image, eval_image):
    mse = np.mean((target_image - eval_image) ** 2)
    if mse == 0:
        return 100
    else:
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#eval_image = Image.open(r"C:\Users\DELL\Desktop\result_collection\17_eval.png")

#eval_image_array = np.array(eval_image)

#target_image = Image.open(r"C:\Users\DELL\Desktop\result_collection\17_target.png")

#target_image_array = np.array(target_image)

#a = psnr(target_image_array, eval_image_array)
#print(a)

