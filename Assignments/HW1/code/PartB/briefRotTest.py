import cv2
import numpy as np
import matplotlib.pyplot as plt
from PartB.my_BRIEF import briefLite, briefMatch


def get_rotations_stats_bar_graph(path):
    img = cv2.imread(path)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_gray = im_gray / 255
    locs_src, desc_src = briefLite(im_gray)

    height, width = im_gray.shape[:2]
    image_center = (width / 2, height / 2)
    tot_matches = []
    deg_arr = np.array(range(10, 360, 10))

    for deg in deg_arr:
        rotation_mat = cv2.getRotationMatrix2D(image_center, deg, 1.)

        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # calc width and height
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_img = cv2.warpAffine(im_gray, rotation_mat, (bound_w, bound_h))

        _, desc_dst = briefLite(rotated_img)
        matches = briefMatch(desc_dst, desc_src)
        matches_size = matches.shape[0]
        tot_matches.append(matches_size)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(deg_arr, tot_matches, width=7)
    ax.set_xlabel('degree of rotation')
    ax.set_ylabel('number of matches')
    ax.grid()


