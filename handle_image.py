from PIL import Image
import numpy as np

def crop_random_subimage(image_path, subimage_size, x, y):
    image = Image.open(image_path)

    width, height = image.size

    if width < subimage_size or height < subimage_size:
        return None

    cropped_image = image.crop((x, y, x + subimage_size, y + subimage_size))

    np_array = np.array(cropped_image)

    return np_array



