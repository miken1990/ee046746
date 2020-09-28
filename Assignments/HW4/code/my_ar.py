import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy

from matplotlib import pyplot as plt
plt.ion()

#Add imports if needed:
import time
from frame_video_convert import video_to_image_seq, image_seq_to_video
import os
#end imports

#Add functions here:
"""
   Your code here
"""

def getPointsOneImage(im, N):
    plt.figure(1)
    plt.imshow(im)
    p = plt.ginput(N, timeout=60)
    p_x = [p[0] for p in p]
    p_y = [p[1] for p in p]
    p_out = np.stack([p_x, p_y])
    return p_out


def im2im(ref_im, plant_im, dest_im, with_bgd=False):
    start = time.time()
    #TODO: match between ref_im and dest_im and calculate homography
    sift = cv2.xfeatures2d.SIFT_create()
    gray_ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    gray_dest_im = cv2.cvtColor(dest_im, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray_ref_im, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray_dest_im, None)
    bf = cv2.BFMatcher(cv2.NORM_L1)
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # calculate max width and max height in dest image from the descriptors
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    plant_im_transformed = cv2.warpPerspective(plant_im, M, (dest_im.shape[1], dest_im.shape[0]))
    plant_im_transformed_mask = cv2.warpPerspective(np.ones(plant_im.shape), M, (dest_im.shape[1], dest_im.shape[0]))
    im_with_plant = dest_im.copy()

    for i in range(3):
        if not with_bgd:
            im_with_plant[:, :, i][plant_im_transformed_mask[:, :, i] > 0] = 0
        im_with_plant[:, :, i][plant_im_transformed[:, :, i] > 0] = plant_im_transformed[:, :, i][plant_im_transformed[:, :, i] > 0]
    plt.figure(3)
    plt.imshow(im_with_plant)
    plt.show()
    end = time.time()
    # plt.pause(30)
    print(f'execution took {end - start}')
    return im_with_plant


def my_vid2vid(ref_im, orig_frame_path, image_to_plant_path, planted_im_path,
               out_video_path='my_data/3.3/vid_with_plant.mp4', with_bgd=False):

    images_orig = os.listdir(orig_frame_path)
    images_plant = os.listdir(image_to_plant_path)

    for im_orig_name, im_plant_name in zip(images_orig, images_plant):
        im_orig_path = os.path.join(orig_frame_path, im_orig_name)
        im_plant_path = os.path.join(image_to_plant_path, im_plant_name)

        plant_im = cv2.imread(im_plant_path)
        plant_im = cv2.resize(src=plant_im, dsize=(ref_im.shape[1], ref_im.shape[0]))
        dest_im = cv2.imread(im_orig_path)
        im_with_plant = im2im(ref_im, plant_im, dest_im, with_bgd)
        cv2.imwrite(os.path.join(planted_im_path, im_orig_name), im_with_plant)

    image_seq_to_video(imgs_path=planted_im_path, output_path=out_video_path)
    return

#Functions end

# HW functions:
def create_ref(im_path):
    """
       Your code here
    """
    #Load image
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # get points in the following order: t1, tr, br, bl
    points = getPointsOneImage(im_gray, 4).astype(dtype='float32')


    (tl, tr, br, bl) = points.T
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(points.T, dst)
    ref_image = cv2.warpPerspective(im, M, (maxWidth, maxHeight))
    plt.figure(2)
    plt.imshow(ref_image)
    plt.show()

    return ref_image


if __name__ == '__main__':
    print('my_ar')
    # 3.1
    ref_im_path = 'data/pf_floor.jpg'
    ref_im = create_ref(ref_im_path)
    cv2.imwrite('my_data/3.1/ref_im.jpeg', ref_im)
    # 3.2
    plant_im_path = 'my_data/economist.jpg'
    plant_im = cv2.imread(plant_im_path)
    plant_im = cv2.resize(plant_im, (ref_im.shape[1], ref_im.shape[0]))

    # create first image plant
    dst_im1_path = 'data/pf_floor_rot.jpg'
    dst_im1 = cv2.imread(dst_im1_path)
    print(f'before im2im')
    im1_with_plant = im2im(ref_im, plant_im, dst_im1)
    cv2.imwrite('my_data/3.2/pf_floor_rot_with_plant.jpg', im1_with_plant)

    # create second image plant
    dst_im2_path = 'data/pf_pile.jpg'
    dst_im2 = cv2.imread(dst_im2_path)
    print(f'before im2im')
    im2_with_plant = im2im(ref_im, plant_im, dst_im2)
    cv2.imwrite('my_data/3.2/pf_pile_with_plant.jpg', im2_with_plant)

    # create third image plant
    dst_im3_path = 'data/pf_desk.jpg'
    dst_im3 = cv2.imread(dst_im3_path)
    print(f'before im2im')
    im3_with_plant = im2im(ref_im, plant_im, dst_im3)
    cv2.imwrite('my_data/3.2/pf_desk_with_plant.jpg', im3_with_plant)

    # 3.3
    ref_im_path = 'my_data/3.3/ref_image.jpeg'
    ref_im = create_ref(ref_im_path)
    # create output folders with images of frames
    orig_vid_path = 'my_data/3.3/videos/video_new.mp4'
    plant_vid_path = 'my_data/3.3/videos/dancing_man_model.mp4'
    out_orig_path = 'my_data/3.3/orig_vid_frames'
    out_plant_path = 'my_data/3.3/vid_with_plant_frames'
    video_to_image_seq(vid_path=orig_vid_path, output_path=out_orig_path)
    image_plant_path = 'my_data/3.3/plant_vid_frames'
    video_to_image_seq(vid_path=plant_vid_path, output_path=image_plant_path)

    my_vid2vid(ref_im=ref_im, orig_frame_path=out_orig_path, image_to_plant_path=image_plant_path,
               planted_im_path=out_plant_path)


