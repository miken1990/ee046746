import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from frame_video_convert import video_to_image_seq, image_seq_to_video
from my_ar import my_vid2vid, create_ref

# lower_bound_fgd = (50, 200, 0)
# upper_bound_fgd = (80, 255, 20)

def _calc_mask(im_path, lower_bound_fgd=(0, 140, 10), upper_bound_fgd=(130, 255, 35),
              in_range=True, is_erode=False):
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fgd_ann = cv2.inRange(img, lower_bound_fgd, upper_bound_fgd)//255
    if not in_range:
        fgd_ann = abs(1 - fgd_ann)
    if is_erode:
        kernel = np.ones((7, 7), np.uint8)
        fgd_ann = cv2.erode(fgd_ann, kernel, iterations=1)

    # im_with_mask = np.copy(img)
    # for i in range(3):
    #     img[:, :, i] = img[:, :, i] * fgd_ann
    img = img * fgd_ann[:, :, np.newaxis]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return fgd_ann, img


def calc_th_masks(in_dir_path, mask_out_dir_path, im_output_dir_path):
    for input_im in os.listdir(in_dir_path):
        im_path = osp.join(in_dir_path, input_im)
        mask, out_im = _calc_mask(im_path, in_range=False, is_erode=True)
        mask_path = osp.join(mask_out_dir_path, input_im)
        cv2.imwrite(mask_path, mask * 255)
        masked_im_path = osp.join(im_output_dir_path, input_im)
        cv2.imwrite(masked_im_path, out_im)

# 3.4
# TODO: pass segmented images, apply threshold to them to remove background and place on book


if __name__ == '__main__':

    # calculate masks and masked images
    in_dir = 'my_data/3.3/plant_vid_frames'
    masks_dir = 'my_data/3.4/masks'
    dancer_with_masks_dir = 'my_data/3.4/with_masks'
    calc_th_masks(in_dir, masks_dir, dancer_with_masks_dir)
    image_seq_to_video(dancer_with_masks_dir, output_path='my_data/3.4/dancing.mp4')
    print(f'finished calculating masks')

    # plant beach image in each dancer image using mask:
    beach_im_path = 'my_data/3.4/bgd_image.jpg'
    beach_im = cv2.imread(beach_im_path)
    dancer_with_beach_path = 'my_data/3.4/dancer_with_beach'
    for idx, file in enumerate(os.listdir(dancer_with_masks_dir)):
        mask_im = cv2.imread(osp.join(masks_dir, file))[:, :, 0]

        dancer_with_masks_im = cv2.imread(osp.join(dancer_with_masks_dir, file))
        if idx == 0:
            beach_im = cv2.resize(src=beach_im, dsize=(dancer_with_masks_im.shape[1], dancer_with_masks_im.shape[0]))

        beach_with_dancer_im = np.copy(beach_im)
        idx = np.where(mask_im > 1)
        beach_with_dancer_im[idx] = dancer_with_masks_im[idx]
        cv2.imwrite(osp.join(dancer_with_beach_path, file), beach_with_dancer_im)
        pass


    # after calculation we ran detectron2 to get pose estimation on images with dancer and beach

    # plant dancer on the beach into our video
    orig_vid_path = 'my_data/3.4/videos/vid_input_tv.mp4'
    out_orig_path = 'my_data/3.4/orig_vid_frames'
    video_to_image_seq(vid_path=orig_vid_path, output_path=out_orig_path)

    image_plant_path = 'my_data/3.4/pose_estimation_output'

    ref_im_rect_path = 'my_data/3.4/ref_image_rect.jpeg'
    if not os.path.isfile(ref_im_rect_path):
        ref_im_path = 'my_data/3.4/ref_image.jpeg'
        ref_im = create_ref(ref_im_path)
        cv2.imwrite('my_data/3.4/ref_image_rect.jpeg', ref_im)
    else:
        ref_im = cv2.imread(ref_im_rect_path)

    out_plant_path = 'my_data/3.4/final_video'

    my_vid2vid(ref_im, orig_frame_path=out_orig_path, image_to_plant_path=image_plant_path,
               planted_im_path=out_plant_path, out_video_path='my_data/3.4/vid_with_plant.mp4', with_bgd=False)






