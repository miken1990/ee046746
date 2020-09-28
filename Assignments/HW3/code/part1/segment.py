import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from PIL import Image


def grab_cut_bbox(im_path, start_point, end_point, mask=None,iters=5):
    im = cv.imread(im_path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = np.array(im)
    if mask is None:
        mask = np.zeros(im.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = start_point + (end_point[0] - start_point[0], end_point[1] - start_point[1])
    mask, bgdModel, fgdModel = cv.grabCut(im, mask, rect, bgd, fgd,iters, cv.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    im = im * mask[:, :, np.newaxis]
    plt.imshow(im), plt.colorbar(), plt.show()
    rec_img = cv.rectangle(im, start_point, end_point, (250, 0, 0), 2)
    cv.imshow('image', rec_img)


def calc_mask(im_path, lower_bound_fgd=(241, 88, 210), upper_bound_fgd=(261, 108, 230),
              in_range=True, is_erode=False):
    img = cv.imread(im_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    fgd_ann = cv.inRange(img, lower_bound_fgd, upper_bound_fgd)//255
    if not in_range:
        fgd_ann = abs(1 - fgd_ann)
    if is_erode:
        kernel = np.ones((7, 7), np.uint8)
        fgd_ann = cv.erode(fgd_ann, kernel, iterations=1)

    return fgd_ann


def pad(array, reference_shape, offsets):
    # Create an array of zeros with the reference shape
    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array

    return result


def calc_masked_image(im_path: str, target_im, shift_down: int, shift_right: int, mask):
    target_shape = (np.array(target_im).shape[0], np.array(target_im).shape[1])
    # create mask with target shape
    padded_mask = pad(mask, target_shape, [0, 0])
    padded_mask = np.roll(padded_mask, [shift_down, shift_right], axis=(0, 1))
    padded_mask = ((1 - padded_mask) * 255).astype(np.uint8)
    mask_for_target = transforms.ToPILImage()(padded_mask).convert('L')
    # shift image
    target_shape = (np.array(target_im).shape[0], np.array(target_im).shape[1], np.array(target_im).shape[2])
    input_image = np.array(Image.open(im_path))
    input_image = pad(input_image, target_shape, [0, 0, 0])
    input_image = np.roll(input_image, [shift_down, shift_right, 0], axis=(0, 1, 2))
    input_image = Image.fromarray(input_image.astype(np.uint8))
    masked_img = Image.composite(target_im, input_image, mask_for_target)
    return masked_img


def segment_deeplabv3(model, filename, is_erode=False):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # perform pre-processing
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of size 1 as expected by the model
    # send to device
    model = model.to(device)
    input_batch = input_batch.to(device)
    # forward pass
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    # print("output shape: ", output.shape)
    # print("output_predictions shape: ", output_predictions.shape)
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    mask = torch.zeros_like(output_predictions).float().to(device)
    mask[output_predictions != 0] = 1 # 12 is dog

    masked_img = input_image * mask.unsqueeze(2).byte().cpu().numpy()
    # fig = plt.figure()
    # fig.set_size_inches(12, 12)

    return mask, masked_img
