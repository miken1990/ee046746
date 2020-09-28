from scipy.io import loadmat
from scipy.spatial.distance import cdist
import numpy as np
from PartA.my_keypoint_det import DoGdetector


def makeTestPattern(patchWidth, nbits):
    """
    Your code here
    """
    set_points = set()
    compareX = []
    compareY = []
    for i in range(nbits):
        p1, p2 =np.random.randint(patchWidth*patchWidth, size=(2))
        while (p1, p2) in set_points or (p2, p1) in set_points or p1 == p2:
            p1, p2 = np.random.randint(patchWidth*patchWidth, size=(2))

        set_points.add((p1, p2))
        compareX.append(p1)
        compareY.append(p2)
    compareX = np.array(compareX)
    compareY = np.array(compareY)
    return compareX, compareY


def computeBrief(im, GaussianPyramid, locsDoG, k, levels,
                 compareX, compareY):
    """
    Your code here
    """
    locs = []
    desc = []

    for kp in locsDoG:

        if kp[0] < 4 or kp[1] < 4 or (kp[1] > GaussianPyramid.shape[1] - 5) \
                or (kp[0] > GaussianPyramid.shape[2] - 5):
            continue
        locs.append(kp)
        # create patch window
        gauss_pyr_idx = levels.index(kp[2])

        patch = GaussianPyramid[gauss_pyr_idx, kp[1] - 4:kp[1] + 5, kp[0] - 4:kp[0] + 5]

        patch = patch.flatten('F')
        patch_desc = []

        for p1, p2 in zip(compareX, compareY):
            indicator = patch[p1] < patch[p2]
            patch_desc.append(indicator)
        desc.append(np.array(patch_desc))
    desc = np.stack(desc)
    locs = np.stack(locs)
    return locs, desc


def briefLite(im):
    """
    Your code here
    """
    sigma0 = 1
    k = np.sqrt(2)
    levels = [-1, 0, 1, 2, 3, 4]
    th_contrast = 0.03
    th_r = 12

    locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, th_contrast, th_r)

    mat_p1_p2 = loadmat('../data/testPattern.mat')
    compareX = mat_p1_p2['compareX'][0]
    compareY = mat_p1_p2['compareY'][0]

    locs, desc = computeBrief(im=None, GaussianPyramid=GaussianPyramid, locsDoG=locsDoG, \
                              k=None, levels=levels, compareX=compareX, compareY=compareY)
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.5):
    #     performs the descriptor matching
    #     inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
    #                                 n is the number of bits in the brief
    #     outputs : matches - p x 2 matrix. where the first column are indices
    #                                         into desc1 and the second column are indices into desc2
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1,ix2), axis=-1)
    return matches
