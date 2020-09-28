import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.io
from matplotlib import pyplot as plt
from scipy import interpolate
import random
import time
np.random.seed(0)

#Add imports if needed:


#end imports

#Add extra functions here:
"""
Your code here
"""
#Extra functions end

# HW functions:
def getPoints(im1,im2,N):
    """
    Your code here
    """
    plt.figure(1)
    plt.imshow(im1)
    p1 = plt.ginput(N, timeout=60)
    p1_x = [p[0] for p in p1]
    p1_y = [p[1] for p in p1]
    p1_out = np.stack([p1_x, p1_y])
    plt.close(1)
    plt.figure(2)
    plt.imshow(im2)
    p2 = plt.ginput(N, timeout=60)
    p2_x = [p[0] for p in p2]
    p2_y = [p[1] for p in p2]
    p2_out = np.stack([p2_x, p2_y])
    plt.close(2)
    return p1_out, p2_out

def computeHAffine(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    b = np.reshape(p2, len(p2.T) * 2, order='F')
    A = np.zeros((2 * p1.shape[1], 6))
    B_1 = np.array([p1[0], p1[1], np.ones(p1.shape[1]).T, np.zeros(p1.shape[1]).T, np.zeros(p1.shape[1]).T,
                    np.zeros(p1.shape[1]).T])
    A[0::2, :] = B_1.T
    B_2 = np.array([np.zeros(p1.shape[1]).T, np.zeros(p1.shape[1]).T, np.zeros(p1.shape[1]).T, p1[0], p1[1],
                    np.ones(p1.shape[1]).T])
    A[1::2, :] = B_2.T
    h = np.linalg.pinv(A.T@A)@(A.T)@b
    h = np.hstack((h,[0,0,1]))
    H2to1 = np.reshape(h,(3,3))
    return H2to1

def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    """
    Your code here
    """
    # H2to1 = cv2.getPerspectiveTransform(p1, p2)
    # construct A matrix
    A = np.zeros((2 * p1.shape[1], 9))
    B_1 = np.array([p1[0], p1[1], np.ones(p1.shape[1]).T, np.zeros(p1.shape[1]).T, np.zeros(p1.shape[1]).T,
                    np.zeros(p1.shape[1]).T, -p1[0] * p2[0], -p1[1] * p2[0], -p2[0]])
    A[0::2, :] = B_1.T
    B_2 = np.array([np.zeros(p1.shape[1]).T, np.zeros(p1.shape[1]).T, np.zeros(p1.shape[1]).T, p1[0], p1[1],
                    np.ones(p1.shape[1]).T, -p1[0] * p2[1], -p1[1] * p2[1], -p2[1]])
    A[1::2, :] = B_2.T
    (U, D, V) = np.linalg.svd(A, False)
    h = V.T[:, -1]
    H2to1 = np.reshape(h, (3, 3))
    H2to1 /= H2to1[2, 2]
    # H2to1, status = cv2.findHomography(p1.T, p2.T)

    return H2to1


def projectPoints(h,im1,im2,N):

    plt.figure(1)
    plt.imshow(im1)
    p1 = plt.ginput(N, timeout=60)
    p1_x = [p[0] for p in p1]
    p1_y = [p[1] for p in p1]
    p1_out = np.stack([p1_x, p1_y])
    p1_homog = np.vstack([p1_out, np.ones((1,N))])
    p2_homog = h@p1_homog;
    p2 = np.stack([p2_homog[0]/p2_homog[2],p2_homog[1]/p2_homog[2]])
    plt.close(1)
    plt.figure(2)
    plt.imshow(im2)
    plt.scatter(x=p2[0], y=p2[1], c='r')
    plt.show()

    pass


def wrapPixel(x,y,h):
    hom = np.array([x,y,1]).T
    transformed = h@hom
    transformed /= transformed[2]
    return transformed


def _get_grid(im, im2_shape, H):
    x_grid = range(im.shape[1])
    y_grid = range(im.shape[0])

    xx, yy = np.meshgrid(x_grid, y_grid)
    # TODO: validate what happens when h is not invertible

    zz = np.ones(xx.shape)
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()])
    # h_inv = np.linalg.pinv(H)
    transformed_points = H @ points
    transformed_points /= transformed_points[2, :]
    min_x_proj = int(np.ceil(np.min(transformed_points[0, :])))
    max_x_proj = int(np.ceil(np.max(transformed_points[0, :])))
    min_y_proj = int(np.ceil(np.min(transformed_points[1, :])))
    max_y_proj = int(np.ceil(np.max(transformed_points[1, :])))

    min_x = min(min_x_proj, 0)
    max_x = max(max_x_proj, im2_shape[1])

    min_y = min(min_y_proj, 0)
    max_y = max(max_y_proj, im2_shape[0])

    if ((max_x-min_x)>5000):
        print("grid too big")
    if ((max_y-min_y)>5000):
        print("grid too big")

    x_grid = range(min_x, max_x)
    y_grid = range(min_y, max_y)
    xx_grid, yy_grid = np.meshgrid(x_grid,y_grid)

    return xx_grid, yy_grid


def warpH(im1, H, im2_shape,inerp_method='linear'):
    """
    Your code here
    # """
    x_grid_in_im1 = range(im1.shape[1])
    y_grid_in_im1 = range(im1.shape[0])
    interp_c0 = interpolate.interp2d(x_grid_in_im1, y_grid_in_im1, im1[:, :, 0], kind=inerp_method, fill_value=0)
    interp_c1 = interpolate.interp2d(x_grid_in_im1, y_grid_in_im1, im1[:, :, 1], kind=inerp_method, fill_value=0)
    interp_c2 = interpolate.interp2d(x_grid_in_im1, y_grid_in_im1, im1[:, :, 2], kind=inerp_method, fill_value=0)
    xx_wraped, yy_wraped =_get_grid(im1, im2_shape, H)


    xx_wraped_flat = xx_wraped.flatten()
    yy_wraped_flat = yy_wraped.flatten()
    warp_im1_0 = np.zeros(xx_wraped_flat.shape)
    warp_im1_1 = np.zeros(xx_wraped_flat.shape)
    warp_im1_2 = np.zeros(xx_wraped_flat.shape)
    h_inv = np.linalg.pinv(H)

    for i in range(warp_im1_0.shape[0]):
        transformed_point = wrapPixel(xx_wraped_flat[i], yy_wraped_flat[i], h_inv)

        if((transformed_point[0] > im1.shape[1])|(transformed_point[1] > im1.shape[0])):
            continue
        warp_im1_0[i] = int(interp_c0(transformed_point[0], transformed_point[1]))
        warp_im1_1[i] = int(interp_c1(transformed_point[0], transformed_point[1]))
        warp_im1_2[i] = int(interp_c2(transformed_point[0], transformed_point[1]))

    warp_im1_0 = np.reshape(warp_im1_0,(xx_wraped.shape[0], xx_wraped.shape[1]))
    warp_im1_1 = np.reshape(warp_im1_1, (xx_wraped.shape[0], xx_wraped.shape[1]))
    warp_im1_2 = np.reshape(warp_im1_2, (xx_wraped.shape[0], xx_wraped.shape[1]))

    warp_im1 = np.dstack((warp_im1_0,warp_im1_1,warp_im1_2))
    warp_im1 = warp_im1.astype(np.uint8)
    # warp_im1 = np.dstack((warp_im1_0,warp_im1_1,warp_im1_2))/255.0

    # pl ow()

    return warp_im1, xx_wraped, yy_wraped


def imageStitching(wrap_img1, img2, xx_wraped, yy_wraped):
    panoImg = np.zeros(wrap_img1.shape)
    panoImg = panoImg + wrap_img1
    [x_start, y_start] = np.where((xx_wraped == 0) & (yy_wraped == 0))
    x_end = x_start + img2.shape[0]
    y_end = y_start + img2.shape[1]
    panoImg[x_start[0]:x_end[0], y_start[0]:y_end[0], :] = img2[:]
    panoImg = panoImg.astype(np.uint8)
    return panoImg



def isInlier(pt1,pt2,H,tresh):
    pt1_H = wrapPixel(pt1[0],pt1[1],H)
    error = np.array((pt1_H[0] - pt2[0],pt1_H[1] - pt2[1]))
    d = np.linalg.norm(error)
    if (d< tresh):
        return 1
    else:
        return 0

def ransacH(locs1, locs2, nIter, tol,isAffine=0):
    locs1_size = locs1.shape[1]
    best_inlier = 0
    bestH=[]
    for i in range (nIter):
        if(isAffine):
            pts_idx = [random.randint(0, locs1_size-1) for p in range(0, 3)]
        else:
            pts_idx = [random.randint(0, locs1_size - 1) for p in range(0, 4)]
        locs1_sampled = locs1[:,pts_idx]
        locs2_sampled = locs2[:, pts_idx]
        if (isAffine==1):
            H = computeHAffine(locs1_sampled, locs2_sampled)
        else:
            H = computeH(locs1_sampled,locs2_sampled)
        inlier_num = 0
        for j in range(locs1_size-1):
            inlier_num += isInlier(locs1[:,j],locs2[:,j],H,tol)
        if (inlier_num > best_inlier):
            bestH = H
            best_inlier = inlier_num
    return bestH,best_inlier

def getPoints_SIFT(im1,im2,N=4,th = 0.8):
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)
    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    good = []
    for m, n in matches:
        if (m.distance < th * n.distance):
            good.append([m])
    img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3), plt.show()
    p1_pos = [[kp.pt[0],kp.pt[1] ]for kp in kp1]
    p2_pos = [[kp.pt[0], kp.pt[1]] for kp in kp2]
    p1_list = [p1_pos[good[i][0].queryIdx] for i in range(len(good))]
    p2_list = [p2_pos[good[i][0].trainIdx] for i in range(len(good))]
    p1 = np.array(p1_list)
    p2 = np.array(p2_list)
    p1 = p1.T
    p2 = p2.T
    if (N > len(p1.T)):
        N= len(p1.T)
    p1 = p1[:, 0:N]
    p2 = p2[:, 0:N]
    # fig = plt.figure(5)
    # ax1 = fig.add_subplot(1,2,1);ax1.imshow(im1);plt.title('im1')
    # ax2 = fig.add_subplot(1, 2, 2);ax2.imshow(im2);plt.title('im2')
    # ax1.scatter(x=p1[0], y=p1[1], c='r')
    # ax2.scatter(x=p2[0], y=p2[1], c='r')
    # plt.show()
    # plt.close(5)
    return p1,p2


def projectInclinePoints():
    im1 = cv2.imread('data/incline_L.png')
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread('data/incline_R.png')
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1_lab = cv2.cvtColor(im1, cv2.COLOR_BGR2LAB)
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im2_shape = im2_gray.shape
    p1_out, p2_out = getPoints(im1_gray, im2_gray, 8)
    H = computeH(p1_out, p2_out)
    projectPoints(H, im1_rgb, im2_rgb, N=6)

def runPanoOnIncline(isRansac,isManual):
    im1 = cv2.imread('data/incline_L.png')
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread('data/incline_R.png')
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # im1_lab = cv2.cvtColor(im1, cv2.COLOR_BGR2LAB)
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im2_shape = im2_gray.shape
    print("FINISHED LOADING IMAGES")
    if(isManual):
        p1_out, p2_out = getPoints(im1_gray, im2_gray, 8)
    else :
        p1_out,p2_out = getPoints_SIFT(im1_gray, im2_gray,N=50)
    if(isRansac):
        H ,best_inlier = ransacH(p1_out, p2_out, nIter=1000, tol=5)
    else:
        H = computeH(p1_out, p2_out)
    print("FINISHED CALCULATINGG H")
    warp_im1, xx_wraped, yy_wraped = warpH(im1_rgb, H, im2_shape, inerp_method='linear')
    print("FINISHED WARPING")
    panoImg = imageStitching(warp_im1, im2_rgb, xx_wraped, yy_wraped)
    print("FINISHED STITCHING")
    panoImg_bgr = cv2.cvtColor(panoImg, cv2.COLOR_RGB2BGR)
    if(isManual):
        cv2.imwrite('my_data/Incline/incline_pano_manual.png', panoImg_bgr)
    else:
        cv2.imwrite('my_data/Incline/incline_pano_sift.png', panoImg_bgr)

def runPanoOnSintra(isRansac,isManual,N,nIter=1000,tol=4):

    im1 = cv2.imread('data/sintra1.jpg')
    # im1 = cv2.resize(im1, (int(im1.shape[0] / 4), int(im1.shape[1] / 4)))
    im1 = cv2.resize(im1, (400, 300))
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    print('finished loading im1')

    im2 = cv2.imread('data/sintra2.jpg')
    # im2 = cv2.resize(im2, (int(im2.shape[0] / 4), int(im2.shape[1] / 4)))
    im2 = cv2.resize(im2, (400, 300))
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    print('finished loading im2')



    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(im1_rgb, im2_rgb, N=N, th=0.6)
        H ,best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(im1_rgb, im2_rgb, N=N, th=0.6)
        H = computeH(p1_sift, p2_sift)
    dst = cv2.warpPerspective(im1_rgb,H, (im2_rgb.shape[0],im2_rgb.shape[1]))
    # plt.figure()
    # plt.imshow(dst)
    # plt.show()
    print('finished ransacH')
    warp_im1, xx_wraped, yy_wraped = warpH(im1_rgb, H, im2_rgb.shape)
    print('finished warpH')
    panoImg0 = imageStitching(warp_im1, im2_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg0_bgr = cv2.cvtColor(panoImg0, cv2.COLOR_RGB2BGR)
    if (isRansac):
        cv2.imwrite('my_data/Sintra/RANSAC_sintra1_sintra2.jpg', panoImg0_bgr)
        scipy.io.savemat('my_data/Sintra/test_warp2to1.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})
    else:
        cv2.imwrite('my_data/Sintra/REG_sintra1_sintra2.jpg', panoImg0_bgr)
        scipy.io.savemat('my_data/Sintra/REG_test_warp2to1.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})

    if(isRansac):
        panoImg0_bgr = cv2.imread('my_data/Sintra/RANSAC_sintra1_sintra2.jpg')
        panoImg0 = cv2.cvtColor(panoImg0_bgr, cv2.COLOR_BGR2RGB)
    else:
        panoImg0_bgr = cv2.imread('my_data/Sintra/REG_sintra1_sintra2.jpg')
        panoImg0 = cv2.cvtColor(panoImg0_bgr, cv2.COLOR_BGR2RGB)

    im3 = cv2.imread('data/sintra3.jpg')
    # im3 = cv2.resize(im3, (int(im3.shape[0] / 4), int(im3.shape[1] / 4)))
    im3 = cv2.resize(im3, (300, 400))
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    print('finished loading im3')



    print('finished getPoints_SIFT')
    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(panoImg0, im3_rgb, N=N, th=0.6)
        H, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(panoImg0, im3_rgb, N=10, th=0.6)
        H = computeH(p1_sift, p2_sift)
    print('finished ransacH')
    panoImg0_warp, xx_wraped, yy_wraped = warpH(panoImg0, H, im3_rgb.shape)
    print('finished warpH')
    panoImg1 = imageStitching(panoImg0_warp, im3_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg1_bgr = cv2.cvtColor(panoImg1, cv2.COLOR_RGB2BGR)
    if (isRansac):
        cv2.imwrite('my_data/Sintra/RANSAC_sintra1_sintra2_sintra3.jpg', panoImg1_bgr)
        scipy.io.savemat('my_data/Sintra/test_warp21to3.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})

    else:
        cv2.imwrite('my_data/Sintra/REG_sintra1_sintra2_sintra3.jpg', panoImg1_bgr)
        scipy.io.savemat('my_data/Sintra/REG_test_warp21to3.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})



    im4 = cv2.imread('data/sintra4.jpg')
    # im4 = cv2.resize(im4, (int(im4.shape[0] / 4), int(im4.shape[1] / 4)))
    im4 = cv2.resize(im4, (300, 400))
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    print('finished loading im4')

    im5 = cv2.imread('data/sintra5.jpg')
    # im5 = cv2.resize(im5, (int(im5.shape[0] / 4), int(im5.shape[1] / 4)))
    im5 = cv2.resize(im5, (300, 400))
    im5_rgb = cv2.cvtColor(im5, cv2.COLOR_BGR2RGB)
    print('finished loading im5')


    print('finished getPoints_SIFT')
    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(im5_rgb, im4_rgb, N=N, th=0.3)
        H, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(im5_rgb, im4_rgb, N=N, th=0.3)
        H = computeH(p1_sift, p2_sift)

    print('finished ransacH')
    im5_warp, xx_wraped, yy_wraped = warpH(im5_rgb, H, im4_rgb.shape)
    print('finished warpH')
    panoImg2 = imageStitching(im5_warp, im4_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg2_bgr = cv2.cvtColor(panoImg2, cv2.COLOR_RGB2BGR)
    if (isRansac):
        cv2.imwrite('my_data/Sintra/RANSAC_sintra5_sintra4.jpg', panoImg2_bgr)
        scipy.io.savemat('my_data/Sintra/test_warp5to4.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})

    else:
        cv2.imwrite('my_data/Sintra/REG_sintra5_sintra4.jpg', panoImg2_bgr)
        scipy.io.savemat('my_data/Sintra/REG_test_warp5to4.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})

    if(isRansac):
        panoImg2_bgr = cv2.imread('my_data/Sintra/RANSAC_sintra5_sintra4.jpg')
        panoImg2 = cv2.cvtColor(panoImg2_bgr, cv2.COLOR_BGR2RGB)
        panoImg1_bgr = cv2.imread('my_data/Sintra/RANSAC_sintra1_sintra2_sintra3.jpg')
        panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)
    else:
        panoImg2_bgr = cv2.imread('my_data/Sintra/REG_sintra5_sintra4.jpg')
        panoImg2 = cv2.cvtColor(panoImg2_bgr, cv2.COLOR_BGR2RGB)
        panoImg1_bgr = cv2.imread('my_data/Sintra/REG_sintra1_sintra2_sintra3.jpg')
        panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)


    print('finished getPoints_SIFT')
    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(panoImg2, panoImg1, N=N, th=0.5)
        H, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(panoImg2, panoImg1, N=N, th=0.5)
        H = computeH(p1_sift, p2_sift)
    print('finished ransacH')
    panoImg2_warp, xx_wraped, yy_wraped = warpH(panoImg2, H, panoImg1.shape)
    print('finished warpH')
    panoImg3 = imageStitching(panoImg2_warp, panoImg1, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg3_bgr = cv2.cvtColor(panoImg3, cv2.COLOR_RGB2BGR)
    if (isRansac):
        cv2.imwrite('my_data/Sintra/RANSAC_sintra5_sintra4_sintra3_sintra2_sintra1.jpg', panoImg3_bgr)
        scipy.io.savemat('my_data/Sintra/test_warp4321to5.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})

    else:
        cv2.imwrite('my_data/Sintra/REG_sintra5_sintra4_sintra3_sintra2_sintra1.jpg', panoImg3_bgr)
        scipy.io.savemat('my_data/Sintra/REG_test_warp4321to5.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H})
    # plt.imshow(panoImg3)
    # plt.show()


def runPanoOnBeach(isRansac, isManual, N, nIter, tol):
    #
    im1 = cv2.imread('data/beach1.jpg')
    im1 = cv2.resize(im1, (600, 800))
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    print('finished loading im1')
    
    im2 = cv2.imread('data/beach2.jpg')
    im2 = cv2.resize(im2, (600, 800))
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    print('finished loading im2')


    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(im1_rgb, im2_rgb, N=N, th=0.5)
        H12, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(im1_rgb, im2_rgb, N=N, th=0.5)
        H12 = computeH(p1_sift,p2_sift)
    print('finished ransacH')
    warp_im1, xx_wraped, yy_wraped = warpH(im1_rgb, H12, im2_rgb.shape)
    warp_im1_bgr = cv2.cvtColor(warp_im1, cv2.COLOR_RGB2BGR)
    print('finished warpH')
    panoImg0 = imageStitching(warp_im1, im2_rgb, xx_wraped, yy_wraped)
    panoImg0_bgr = cv2.cvtColor(panoImg0, cv2.COLOR_RGB2BGR)
    print('finished imageStitching')
    if (isRansac):
        cv2.imwrite('my_data/Beach/RANSAC_Beach1_Beach2.jpg', panoImg0_bgr)
    else:
        cv2.imwrite('my_data/Beach/REG_Beach1_Beach2.jpg', panoImg0_bgr)

    if (isRansac):
        panoImg0_bgr = cv2.imread('my_data/Beach/RANSAC_Beach1_Beach2.jpg')
        panoImg0 = cv2.cvtColor(panoImg0_bgr, cv2.COLOR_BGR2RGB)
    else:
        panoImg0_bgr = cv2.imread('my_data/Beach/REG_Beach1_Beach2.jpg')
        panoImg0 = cv2.cvtColor(panoImg0_bgr, cv2.COLOR_BGR2RGB)


    im3 = cv2.imread('data/beach3.jpg')
    im3 = cv2.resize(im3, (600, 800))
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    print('finished loading im3')
    #
    p1_sift, p2_sift = getPoints_SIFT(panoImg0, im3_rgb, N=N,th=0.5)
    print('finished getPoints_SIFT')
    if (isRansac):
        H123, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        H123 = computeH(p1_sift,p2_sift)
    print('finished ransacH')
    panoImg0_warp, xx_wraped, yy_wraped = warpH(panoImg0, H123, im3_rgb.shape)
    print('finished warpH')
    panoImg1 = imageStitching(panoImg0_warp, im3_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg1_bgr = cv2.cvtColor(panoImg1, cv2.COLOR_RGB2BGR)
    if(isRansac):
        cv2.imwrite('my_data/Beach/RANSAC_beach1_beach2_beach3.jpg', panoImg1_bgr)
    else:
        cv2.imwrite('my_data/Beach/REG_beach1_beach2_beach3.jpg', panoImg1_bgr)

    if (isRansac):
        panoImg1_bgr = cv2.imread('my_data/Beach/RANSAC_beach1_beach2_beach3.jpg')
        panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)
    else:
        panoImg1_bgr = cv2.imread('my_data/Beach/REG_beach1_beach2_beach3.jpg')
        panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)


    im4 = cv2.imread('data/beach4.jpg')
    im4 = cv2.resize(im4, (600, 800))
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    print('finished loading im4')

    im5 = cv2.imread('data/beach5.jpg')
    im5 = cv2.resize(im5, (600, 800))
    im5_rgb = cv2.cvtColor(im5, cv2.COLOR_BGR2RGB)
    print('finished loading im5')


    print('finished getPoints_SIFT')
    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(im5_rgb, im4_rgb, N=N, th=0.5)
        H45, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(im5_rgb, im4_rgb, N=N, th=0.5)
        H45 = computeH(p1_sift,p2_sift)
    dst = cv2.warpPerspective(im5_rgb, H45, (im4_rgb.shape[0], im4_rgb.shape[1]))
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/Beach/warp45CV.jpg', dst_bgr)
    print('finished ransacH')
    im5_warp, xx_wraped, yy_wraped = warpH(im5_rgb, H45, im4_rgb.shape)
    im5_warp_bgr = cv2.cvtColor(im5_warp, cv2.COLOR_RGB2BGR)
    print('finished warpH')
    panoImg2 = imageStitching(im5_warp, im4_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg2_bgr = cv2.cvtColor(panoImg2, cv2.COLOR_RGB2BGR)
    if(isRansac):
        cv2.imwrite('my_data/Beach/RANSAC_beach5_beach4.jpg', panoImg2_bgr)
    else:
        cv2.imwrite('my_data/Beach/REG_beach5_beach4.jpg', panoImg2_bgr)

    if (isRansac):
        panoImg2_bgr = cv2.imread('my_data/Beach/RANSAC_beach5_beach4.jpg')
        panoImg2 = cv2.cvtColor(panoImg2_bgr, cv2.COLOR_BGR2RGB)
        panoImg1_bgr = cv2.imread('my_data/Beach/RANSAC_beach1_beach2_beach3.jpg')
        panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)
    else:
        panoImg2_bgr = cv2.imread('my_data/Beach/REG_beach5_beach4.jpg')
        panoImg2 = cv2.cvtColor(panoImg2_bgr, cv2.COLOR_BGR2RGB)
        panoImg1_bgr = cv2.imread('my_data/Beach/REG_beach1_beach2_beach3.jpg')
        panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)


    print('finished getPoints_SIFT')
    if (isRansac):
        p1_sift, p2_sift = getPoints_SIFT(panoImg2, panoImg1, N=N, th=0.5)
        H54TO321, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    else:
        p1_sift, p2_sift = getPoints_SIFT(panoImg2, panoImg1, N=N, th=0.3)
        H54TO321 = computeH(p1_sift,p2_sift)
    print('finished ransacH')
    dst = cv2.warpPerspective(panoImg2, H54TO321, (panoImg1.shape[0], panoImg1.shape[1]))
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/Beach/warp54CV.jpg', dst_bgr)
    panoImg2_warp, xx_wraped, yy_wraped = warpH(panoImg2, H54TO321, panoImg1.shape)
    panoImg2_warp_bgr = cv2.cvtColor(panoImg2_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/Beach/warp54.jpg', panoImg2_warp_bgr)
    scipy.io.savemat('my_data/Beach/beach_test_warp54to321.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H54TO321})
    print('finished warpH')
    panoImg3 = imageStitching(panoImg2_warp, panoImg1, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg3_bgr = cv2.cvtColor(panoImg3, cv2.COLOR_RGB2BGR)
    if(isRansac):
        cv2.imwrite('my_data/Beach/RANSAC_beach4_beach3_beach2_beach1.jpg', panoImg3_bgr)
    else:
        cv2.imwrite('my_data/Beach/REG_beach4_beach3_beach2_beach1.jpg', panoImg3_bgr)

def panoMyImg(N,nIter,tol):
    im1 = cv2.imread('my_data\my_view\mv1.jpg')
    im1 = cv2.resize(im1, (600, 800))
    im1_rgb = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)

    im2 = cv2.imread('my_data\my_view\mv2.jpg')
    im2 = cv2.resize(im2, (600, 800))
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    print('finished loading im2')

    p1_sift, p2_sift = getPoints_SIFT(im1_rgb, im2_rgb, N=N)
    H12, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    print('finished ransacH')
    warp_im1, xx_wraped, yy_wraped = warpH(im1_rgb, H12, im2_rgb.shape)
    warp_im1_bgr = cv2.cvtColor(warp_im1, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view/warp12.jpg', warp_im1_bgr)
    scipy.io.savemat('my_data/my_view/test_warp2to1.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H12})
    print('finished warpH')
    panoImg0 = imageStitching(warp_im1, im2_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg0_bgr = cv2.cvtColor(panoImg0, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view/RANSAC_mv1_mv2.jpg', panoImg0_bgr)

    panoImg0_bgr = cv2.imread('my_data/my_view/RANSAC_mv1_mv2.jpg')
    panoImg0 = cv2.cvtColor(panoImg0_bgr, cv2.COLOR_BGR2RGB)

    im3 = cv2.imread('my_data\my_view\mv3.jpg')
    im3 = cv2.resize(im3, (600, 800))
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    print('finished loading im3')
    #
    p1_sift, p2_sift = getPoints_SIFT(panoImg0, im3_rgb, N=N)
    print('finished getPoints_SIFT')
    H23, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    print('finished ransacH')
    panoImg0_warp, xx_wraped, yy_wraped = warpH(panoImg0, H23, im3_rgb.shape)
    panoImg0_warp_bgr = cv2.cvtColor(panoImg0_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\warp23.jpg', panoImg0_warp_bgr)
    scipy.io.savemat('my_data/beach_test_warp2to3.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H23})
    panoImg0_warp_bgr = cv2.cvtColor(panoImg0_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\RpanoImg0_warp_bgr.jpg', panoImg0_warp_bgr)
    print('finished warpH')
    panoImg1 = imageStitching(panoImg0_warp, im3_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg1_bgr = cv2.cvtColor(panoImg1, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view\RANSAC_beach_mv3_mv2_mv1.jpg', panoImg1_bgr)

    panoImg1_bgr = cv2.imread('my_data/my_view\RANSAC_beach_mv3_mv2_mv1.jpg')
    panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)

    im4 = cv2.imread('my_data\my_view\mv4.jpg')
    im4 = cv2.resize(im4, (600, 800))
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    print('finished loading im4')

    im5 = cv2.imread('my_data\my_view\mv5.jpg')
    im5 = cv2.resize(im5, (600, 800))
    im5_rgb = cv2.cvtColor(im5, cv2.COLOR_BGR2RGB)
    print('finished loading im5')

    p1_sift, p2_sift = getPoints_SIFT(im5_rgb, im4_rgb, N=N,th=0.6)
    print('finished getPoints_SIFT')
    H45, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    dst = cv2.warpPerspective(im5_rgb, H45, (im4_rgb.shape[0], im4_rgb.shape[1]))
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\warp45CV.jpg', dst_bgr)
    print('finished ransacH')
    im5_warp, xx_wraped, yy_wraped = warpH(im5_rgb, H45, im4_rgb.shape)
    im5_warp_bgr = cv2.cvtColor(im5_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\warp45.jpg', im5_warp_bgr)
    scipy.io.savemat('my_data\my_view\mv_test_warp5to4.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H45})
    print('finished warpH')
    panoImg2 = imageStitching(im5_warp, im4_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg2_bgr = cv2.cvtColor(panoImg2, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\RANSAC_mv4_mv5.jpg', panoImg2_bgr)

    panoImg2_bgr = cv2.imread('my_data\my_view\RANSAC_mv4_mv5.jpg')
    panoImg2 = cv2.cvtColor(panoImg2_bgr, cv2.COLOR_BGR2RGB)

    p1_sift, p2_sift = getPoints_SIFT(panoImg2, panoImg1, N=8000,th=0.5)
    print('finished getPoints_SIFT')
    H54TO321, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    print('finished ransacH')
    dst = cv2.warpPerspective(panoImg2, H54TO321, (panoImg1.shape[0], panoImg1.shape[1]))
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\warp54to321CV.jpg', dst_bgr)
    panoImg2_warp, xx_wraped, yy_wraped = warpH(panoImg2, H54TO321, panoImg1.shape)
    panoImg2_warp_bgr = cv2.cvtColor(panoImg2_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\warp54.jpg', panoImg2_warp_bgr)
    scipy.io.savemat('my_data\my_view\mv_test_warp54to321.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H54TO321})
    print('finished warpH')
    panoImg3 = imageStitching(panoImg2_warp, panoImg1, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg3_bgr = cv2.cvtColor(panoImg3, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view\RANSAC_mv5_mv4_mv3_mv2_mv1.jpg', panoImg3_bgr)


def panoMyImgAffine(N,nIter,tol):
    im4 = cv2.imread('my_data\my_view_affine\mv4.jpg')
    im4 = cv2.resize(im4, (600, 800))
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)

    im3 = cv2.imread('my_data\my_view_affine\mv3.jpg')
    im3 = cv2.resize(im3, (600, 800))
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    print('finished loading im3')

    p1_sift, p2_sift = getPoints_SIFT(im4_rgb, im3_rgb, N=N)
    H43, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol,isAffine=1)
    print('finished ransacH')
    warp_im4, xx_wraped, yy_wraped = warpH(im4_rgb, H43, im3_rgb.shape)
    warp_im4_bgr = cv2.cvtColor(warp_im4, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view_affine/warp12.jpg', warp_im4_bgr)
    scipy.io.savemat('my_data/my_view_affine/test_warp2to1.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H43})
    print('finished warpH')
    panoImg0 = imageStitching(warp_im4, im3_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg0_bgr = cv2.cvtColor(panoImg0, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view_affine/RANSAC_mv1_mv2.jpg', panoImg0_bgr)

    panoImg0_bgr = cv2.imread('my_data/my_view_affine/RANSAC_mv1_mv2.jpg')
    panoImg0 = cv2.cvtColor(panoImg0_bgr, cv2.COLOR_BGR2RGB)

    im2 = cv2.imread('my_data\my_view_affine\mv2.jpg')
    im2 = cv2.resize(im2, (600, 800))
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    print('finished loading im3')
    #
    p1_sift, p2_sift = getPoints_SIFT(panoImg0, im2_rgb, N=N)
    print('finished getPoints_SIFT')
    H432, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    print('finished ransacH')
    panoImg0_warp, xx_wraped, yy_wraped = warpH(panoImg0, H432, im2_rgb.shape)
    panoImg0_bgr = cv2.cvtColor(panoImg0_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view_affine\warp23.jpg', panoImg0_bgr)
    scipy.io.savemat('my_data/my_view_affine/_test_warp2to3.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H432})
    print('finished warpH')
    panoImg1 = imageStitching(panoImg0_warp, im2_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg1_bgr = cv2.cvtColor(panoImg1, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view_affine\RANSAC_beach_mv3_mv2_mv1.jpg', panoImg1_bgr)

    panoImg1_bgr = cv2.imread('my_data/my_view_affine\RANSAC_beach_mv3_mv2_mv1.jpg')
    panoImg1 = cv2.cvtColor(panoImg1_bgr, cv2.COLOR_BGR2RGB)

    im1 = cv2.imread('my_data\my_view_affine\mv1.jpg')
    im1 = cv2.resize(im1, (600, 800))
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    print('finished loading im1')

    p1_sift, p2_sift = getPoints_SIFT(panoImg1, im1_rgb, N=N)
    print('finished getPoints_SIFT')
    H4321, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol)
    print('finished ransacH')
    panoImg1_warp, xx_wraped, yy_wraped = warpH(panoImg1, H4321, im1_rgb.shape)
    panoImg1_bgr = cv2.cvtColor(panoImg1_warp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data\my_view_affine\warp432.jpg', panoImg1_bgr)
    scipy.io.savemat('my_data/my_view_affine/_test_warp432to1.mat', {'xx_wraped': xx_wraped, 'yy_wraped': yy_wraped, 'H': H4321})
    print('finished warpH')
    panoImg1 = imageStitching(panoImg1_warp, im1_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg1_bgr = cv2.cvtColor(panoImg1, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view_affine\RANSAC_mv4_mv3_mv2_mv1.jpg', panoImg1_bgr)

def panoMyImgAffineWrongImages(N,nIter,tol):
    im1 = cv2.imread('my_data\my_view\mv1.jpg')
    im1 = cv2.resize(im1, (600, 800))
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    print('finished loading im1')

    im2 = cv2.imread('my_data\my_view\mv2.jpg')
    im2 = cv2.resize(im2, (600, 800))
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    print('finished loading im3')

    p1_sift, p2_sift = getPoints_SIFT(im2_rgb, im1_rgb, N=N)
    print('finished getPoints_SIFT')
    H21, best_inlier = ransacH(p1_sift, p2_sift, nIter=nIter, tol=tol,isAffine=1)
    print('finished ransacH')
    im2_rgb_warp, xx_wraped, yy_wraped = warpH(im2_rgb, H21, im1_rgb.shape)

    print('finished warpH')
    panoImg1 = imageStitching(im2_rgb_warp, im1_rgb, xx_wraped, yy_wraped)
    print('finished imageStitching')
    panoImg1_bgr = cv2.cvtColor(panoImg1, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_data/my_view\mv2_mv1_Affine.jpg', panoImg1_bgr)



if __name__ == '__main__':
    projectInclinePoints()
    print("finished: projectInclinePoints()")
    runPanoOnIncline(isRansac=0, isManual=1)
    print("finished: runPanoOnIncline(isRansac=0, isManual=1)")
    runPanoOnIncline(isRansac=0, isManual=0)
    print("finished: runPanoOnIncline(isRansac=0, isManual=0)")
    runPanoOnIncline(isRansac=1, isManual=0)
    print("finished: runPanoOnIncline(isRansac=1, isManual=0)")
    runPanoOnSintra(isRansac=0, isManual=0,N=8,nIter=1000,tol=5)
    print("finished: runPanoOnSintra(isRansac=0, isManual=0,N=8,nIter=1000,tol=5)")
    runPanoOnSintra(isRansac=1, isManual=0, N=200, nIter=5000, tol=4)
    print("finished: runPanoOnSintra(isRansac=1, isManual=0, N=200, nIter=5000, tol=4)")
    runPanoOnBeach(isRansac=1, isManual=0, N=200, nIter=5000, tol=4)
    print("finished: runPanoOnBeach(isRansac=1, isManual=0, N=200, nIter=5000, tol=4)")
    runPanoOnBeach(isRansac=0, isManual=0, N=8, nIter=1000, tol=4)
    print("finished: runPanoOnBeach(isRansac=0, isManual=0, N=8, nIter=1000, tol=4)")
    panoMyImg( N=500, nIter=5000, tol=4)
    print("finished: panoMyImg( N=500, nIter=5000, tol=4)")
    panoMyImgAffine(N=500, nIter=2000, tol=4)
    print("finished: panoMyImgAffine(N=500, nIter=2000, tol=4)")
    panoMyImgAffineWrongImages(N=500, nIter=2000, tol=4)
    print("finished: panoMyImgAffineWrongImages(N=500, nIter=2000, tol=4)")

