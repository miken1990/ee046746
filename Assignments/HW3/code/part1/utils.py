import matplotlib.pyplot as plt
import os
import cv2 as cv
from PIL import Image


def load_image(path):
    image = Image.open(path)
    return image


def resize_and_save(path, row, col):
    image = Image.open(path).resize((col, row))
    image.save(path)


def show_images(image_dir_path):
    fig = plt.figure(figsize=(10, 10))
    for i, im_name in enumerate(os.listdir(image_dir_path)):
        full_im_path = os.path.join(image_dir_path, im_name)
        im = load_image(full_im_path)
        ax = fig.add_subplot(1, len(os.listdir(image_dir_path)), i+1) # create a subplot of certain size
        ax.imshow(im)
        ax.set_title(f'{im_name}')
        ax.set_axis_off()


def show_im_with_bbox(im_path, start_point, end_point):
    im_cv = cv.imread(im_path)
    im_cv = cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)
    rec_img = cv.rectangle(im_cv, start_point, end_point, (250, 0, 0), 2)
    plt.imshow(rec_img), plt.colorbar(), plt.show()
    return rec_img