import numpy as np
import matplotlib.pyplot as plt
import cv2


def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor( 3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im,(size,size),sigma_)
        GaussianPyramid.append(blur)

    res = np.stack(GaussianPyramid)
    return res


def displayPyramid(pyramid):
    plt.figure(figsize=(16,5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')


def createDoGPyramid(GaussianPyramid, levels):
    # Produces DoG Pyramid
    # inputs
    # Gaussian Pyramid - A matrix of grayscale images of size
    #                    (len(levels), shape(im))
    # levels      - the levels of the pyramid where the blur at each level is
    #               outputs
    # DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #               created by differencing the Gaussian Pyramid input
    """
    Your code here
    """
    DoGPyramid = np.diff(GaussianPyramid, axis=0)
    DoGLevels = levels[:-1]
    return DoGPyramid, DoGLevels


def computePrincipalCurvature(DoGPyramid):
    # Edge Suppression
    #  Takes in DoGPyramid generated in createDoGPyramid and returns
    #  PrincipalCurvature,a matrix of the same size where each point contains the
    #  curvature ratio R for the corre-sponding point in the DoG pyramid
    #
    #  INPUTS
    #  DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    #  OUTPUTS
    #  PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    #                       point contains the curvature ratio R for the
    #                       corresponding point in the DoG pyramid
    """
    Your code here
    """
    PrincipalCurvature = []
    for dog in DoGPyramid:
        Dx = np.abs(cv2.Sobel(dog, cv2.CV_64F, 1, 0, ksize=3, ))
        Dy = np.abs(cv2.Sobel(dog, cv2.CV_64F, 0, 1, ksize=3, ))
        Dxx = np.abs(cv2.Sobel(Dx, cv2.CV_64F, 1, 0, ksize=3, ))
        Dxy = np.abs(cv2.Sobel(Dx, cv2.CV_64F, 0, 1, ksize=3, ))
        Dyy = np.abs(cv2.Sobel(Dy, cv2.CV_64F, 0, 1, ksize=3, ))
        TR_H_2 = ((Dxx + Dyy) ** 2)
        Det_h = (Dxx * Dyy - Dxy ** 2)
        R = TR_H_2 / Det_h
        PrincipalCurvature.append(R)

    #         plt.figure(figsize=(16,4))
    #         plt.subplot(1,5,1)
    #         plt.title('Dx')
    #         plt.imshow(Dx,cmap='gray')
    #         plt.axis('off')

    #         plt.subplot(1,5,2)
    #         plt.title('Dy')
    #         plt.imshow(Dy,cmap='gray')
    #         plt.axis('off')

    #         plt.subplot(1,5,3)
    #         plt.title('Dxx')
    #         plt.imshow(Dxx,cmap='gray')
    #         plt.axis('off')

    #         plt.subplot(1,5,4)
    #         plt.title('Dyy')
    #         plt.imshow(Dyy,cmap='gray')
    #         plt.axis('off')

    #         plt.subplot(1,5,5)
    #         plt.title('Dxy')
    #         plt.imshow(Dxy,cmap='gray')
    #         plt.axis('off')

    PrincipalCurvature = np.stack(PrincipalCurvature)
    return PrincipalCurvature


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,
                    th_contrast, th_r):
    #     Returns local extrema points in both scale and space using the DoGPyramid
    #     INPUTS
    #         DoG_pyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    #         DoG_levels  - The levels of the pyramid where the blur at each level is
    #                       outputs
    #         principal_curvature - size (len(levels) - 1, imH, imW) matrix contains the
    #                       curvature ratio R
    #         th_contrast - remove any point that is a local extremum but does not have a
    #                       DoG response magnitude above this threshold
    #         th_r        - remove any edge-like points that have too large a principal
    #                       curvature ratio
    #      OUTPUTS
    #         locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
    #                scale and space, and also satisfies the two thresholds.

    """
    Your code here
    """

    def is_biger_then_neighbours(i, j, pad_dog):
        #         if (i==2 and j==2 and pad_dog[i,j] == 2) or (i==4 and j==4 and pad_dog[i,j] == 5):
        #             print('\n')
        #             print(f'[{pad_dog[i-1, j-1]}, {pad_dog[i-1, j]}, {pad_dog[i-1, j+1]}]')
        #             print(f'[{pad_dog[i, j-1]}, {pad_dog[i, j]}, {pad_dog[i, j+1]}]')
        #             print(f'[{pad_dog[i+1, j-1]}, {pad_dog[i+1, j]}, {pad_dog[i+1, j+1]}]')

        return pad_dog[i, j] > pad_dog[i - 1, j - 1] and pad_dog[i, j] > pad_dog[i - 1, j] and pad_dog[i, j] > pad_dog[
            i - 1, j + 1] and \
               pad_dog[i, j] > pad_dog[i, j - 1] and pad_dog[i, j] > pad_dog[i, j + 1] and \
               pad_dog[i, j] > pad_dog[i + 1, j - 1] and pad_dog[i, j] > pad_dog[i + 1, j] and pad_dog[i, j] > pad_dog[
                   i + 1, j + 1]

    def calc_bigger_then_all_neighobours(i, j, p_dog, pad_dog, n_dog):
        is_bigger_then_p_dog = True if p_dog is None else False
        is_bigger_then_n_dog = True if n_dog is None else False

        # check dog
        is_bigger_then_neighbours = is_biger_then_neighbours(i, j, pad_dog)

        if p_dog is not None:
            is_bigger_then_p_dog = pad_dog[i, j] > p_dog[i - 1, j - 1]

        if n_dog is not None:
            is_bigger_then_n_dog = pad_dog[i, j] > n_dog[i - 1, j - 1]

        return is_bigger_then_neighbours and is_bigger_then_p_dog and is_bigger_then_n_dog

    locsDoG = []
    for idx, dog in enumerate(DoGPyramid):
        pad_dog = np.pad(dog, (1, 1), 'constant', constant_values=0)
        p_dog = DoGPyramid[idx - 1] if idx > 0 else None
        n_dog = DoGPyramid[idx + 1] if idx < (DoGPyramid.shape[0] - 1) else None
        for i in range(dog.shape[0]):
            for j in range(dog.shape[1]):
                is_bigger_then_all_neighbours = calc_bigger_then_all_neighobours(i + 1, j + 1, p_dog, pad_dog, n_dog)
                is_bigger_then_theta_c = dog[i, j] > th_contrast
                is_smaller_theta_r = PrincipalCurvature[idx, i, j] < th_r
                if is_bigger_then_all_neighbours and is_bigger_then_theta_c and is_smaller_theta_r:
                    locsDoG.append(np.array([j, i, DoGLevels[idx]]))

    locsDoG = np.stack(locsDoG)
    return locsDoG


def DoGdetector(im, sigma0, k, levels,
    th_contrast, th_r):
    #     Putting it all together
    #     Inputs          Description
    #     --------------------------------------------------------------------------
    #     im              Grayscale image with range [0,1].
    #     sigma0          Scale of the 0th image pyramid.
    #     k               Pyramid Factor.  Suggest sqrt(2).
    #     levels          Levels of pyramid to construct. Suggest -1:4.
    #     th_contrast     DoG contrast threshold.  Suggest 0.03.
    #     th_r            Principal Ratio threshold.  Suggest 12.
    #     Outputs         Description
    #     --------------------------------------------------------------------------
    #     locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
    #                     in both scale and space, and satisfies the two thresholds.
    #     gauss_pyramid   A matrix of grayscale images of size (len(levels),imH,imW)
    """
    Your code here
    """
    GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
    PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
    locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r)

    return locsDoG, GaussianPyramid

