import torchvision
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range - approximately...
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1, 2, 0)


def show_5_images(dl_train, classes) :
    dataiter = iter(dl_train)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
    for idx, image in enumerate(images):
        axes[idx].imshow(convert_to_imshow_format(image))
        axes[idx].set_title(classes[labels[idx].item()])
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.show()




