import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import ntpath
from collections import Counter

def whiteThreshold(img, threshold=60):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)

    v[s < threshold] = 0

    img = cv.merge([h, s, v])
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    return img


def ycrcbThreshold(img, Crlower, Crupper, Cblower, Cbupper):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img_ycrcb)

    img[cr < Crlower] = [0, 0, 0]
    img[cr > Crupper] = [0, 0, 0]
    img[cb < Cblower] = [0, 0, 0]
    img[cb > Cbupper] = [0, 0, 0]

    return img


def otsu(img):
    ot, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU, )
    return img


def ROIotsu(img, vertical_blocks=5, horizontal_blocks=3):
    img_shape = np.shape(img)
    ROI_shape = (math.ceil(img_shape[0] / vertical_blocks), math.ceil(img_shape[1] / horizontal_blocks))

    for i in range(vertical_blocks):
        for j in range(horizontal_blocks):
            ROI = img[i * ROI_shape[0]:(i + 1) * ROI_shape[0], j * ROI_shape[1]:(j + 1) * ROI_shape[1]]
            ot, ROI = cv.threshold(ROI, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU, )
            img[i * ROI_shape[0]:(i + 1) * ROI_shape[0], j * ROI_shape[1]:(j + 1) * ROI_shape[1]] = ROI

    return img


def Erosion(img, erosion_size=3, erosion_shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    img_final = cv.erode(img, element)

    return img_final


def Dilation(img, dilation_size=3, dilation_shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    img_final = cv.dilate(img, element)

    return img_final


def Closing(img, size=3, shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, element)

    return closing


def histogram(img, scale=256):
    hist = cv.calcHist([img], [0], None, [scale], [0, scale])
    return hist


def equalizeHistogram(img):
    b = 255  # white colour intensity
    n = img.size  # number of all pixels
    temp = np.copy(img)  # output image
    s = 0  # number of processed pixels
    for k in range(b + 1):  # pass pixels through all intensities
        w = img == k  # pixels with this intensity 'k'
        p = np.sum(w)  # number of these pixels
        temp[w] = np.round((s + p / 2) * b / n)  # new intensity
        s += p  # add these new processed pixels to the all processed
    return temp


def YCrCbHistogram(img):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

    y, cr, cb = cv.split(img_ycrcb)

    histY = histogram(y)
    histCr = histogram(cr)
    histCb = histogram(cb)

    return histY, histCr, histCb


def plotHistogram(img_ycrcb, name, path, array_components=(1, 2), array_range=(0, 256)):
    upper_ranges = np.ones_like(array_components) * array_range[1]

    ranges = np.zeros((len(array_components) * 2))
    ranges[0::2] = array_range[0]
    ranges[1::2] = array_range[1]

    hist = cv.calcHist([img_ycrcb], array_components, None, upper_ranges, ranges)

    if len(array_components) == 3:
        z, x, y = hist.nonzero()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z, cmap='Greens')
        ax.set_xlabel('$Cr$', fontsize=20)
        ax.set_ylabel('$Cb$', fontsize=20)
        ax.set_zlabel('$Y$', fontsize=20)
        plt.savefig(path + "/histogram" + name + ".jpg")
        plt.show()

    elif len(array_components) == 2:
        x, y = hist.nonzero()
        plt.scatter(x, y, cmap="Greens")
        plt.xlabel('$Cr$', fontsize=20)
        plt.ylabel('$Cb$', fontsize=20)
        plt.savefig(path + "/histogram" + name + ".jpg")
        plt.show()


def Backprojection(img_reference_ycrcb, img_test_ycrcb, threshold=25):
    hist_ref = cv.calcHist([img_reference_ycrcb], [1, 2], None, [256, 256], [0, 256, 0, 256])
    hist_ref[128, 128] = 0

    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(hist_ref, -1, disc, hist_ref)

    cv.normalize(hist_ref, hist_ref, 0, 255, cv.NORM_MINMAX)
    prob = cv.calcBackProject([img_test_ycrcb], [1, 2], hist_ref, [0, 256, 0, 256], 1)

    # disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # cv.filter2D(prob, -1, disc, prob)

    result = np.ones_like(prob)
    result[prob < threshold] = 0

    return result, prob


def plotHistogramYCrCb(img, path, counter, scale=256, name="image"):
    histY, histCr, histCb = YCrCbHistogram(img)

    histY = histY / (np.shape(img)[0] * np.shape(img)[1])
    histCr = histCr / (np.shape(img)[0] * np.shape(img)[1])
    histCb = histCb / (np.shape(img)[0] * np.shape(img)[1])

    plt.subplot(311)
    plt.stem(histY)
    plt.subplot(312)
    plt.stem(histCr)
    plt.subplot(313)
    plt.stem(histCb)
    plt.savefig(path + "/histogram " + name + " " + str(counter) + ".jpg")
    plt.show()

    cv.imwrite(path + "/" + name + " " + str(counter) + ".jpg", img)
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img_ycrcb)
    cv.imwrite(path + "/" + name + "_cr " + str(counter) + ".jpg", cr)
    cv.imwrite(path + "/" + name + "_cb " + str(counter) + ".jpg", cb)


def adaptiveHistogramFilter(face_img, relative_threshold=0.5):
    faceY, faceCr, faceCb = YCrCbHistogram(face_img)

    totalCr = np.sum(faceCr)
    totalCb = np.sum(faceCb)

    maxCr_index = np.argmax(faceCr)
    maxCb_index = np.argmax(faceCb)

    sumCr = 0
    sumCb = 0
    i = 0

    while sumCr < relative_threshold * totalCr:
        sumCr = np.sum(faceCr[maxCr_index - i:maxCr_index + i])
        i += 1

    Crlower = maxCr_index - i
    Crupper = maxCr_index + i

    i = 0

    while sumCb < relative_threshold * totalCb:
        sumCb = np.sum(faceCb[maxCb_index - i:maxCb_index + i])
        i += 1

    Cblower = maxCb_index - i
    Cbupper = maxCb_index + i

    return Crlower, Crupper, Cblower, Cbupper


def drawDetections(img, bbox, hand_window_size_mult=1.6):
    Xupper = bbox[0] - 20 - math.ceil(hand_window_size_mult * bbox[2])
    Yupper = bbox[1] - math.ceil(((hand_window_size_mult - 1) / 2) * bbox[3])
    Xlower = bbox[0] - 20
    Ylower = bbox[1] + math.ceil(((hand_window_size_mult - 1) / 2 + 1) * bbox[3])
    cv.rectangle(img, [bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], (0, 255, 0), 3)
    cv.rectangle(img, [Xupper, Yupper], [Xlower, Ylower], (0, 0, 255), 3)

    return Xupper, Yupper, Xlower, Ylower


def ImageAveraging(img1, img2, buffer, bufferX, bufferY, frames=3):
    buffer.append(img1)
    bufferX.append(np.shape(img1)[1])
    bufferY.append(np.shape(img1)[0])
    xgap = None
    ygap = None
    if len(buffer) > frames:
        maxX = np.max(bufferX)
        maxY = np.max(bufferY)

        avg_image = np.zeros([maxY, maxX])
        for i in range(len(buffer)):
            image_shape = np.shape(buffer[i])
            xgap = math.ceil((maxX - image_shape[1]) / 2)
            ygap = math.ceil((maxY - image_shape[0]) / 2)

            img1_resize = cv.resize(buffer[i], (maxX - 2 * xgap, maxY - 2 * ygap))

            background = np.zeros([maxY, maxX])
            background[ygap:maxY - ygap, xgap:maxX - xgap] = img1_resize

            avg_image = avg_image + background

        avg_image[avg_image < np.amax(avg_image)] = 0
        avg_image = np.clip(avg_image, 0, 1)
        background2 = np.zeros((maxY, maxX, 3), np.uint8)
        img2_resize = cv.resize(img2, (maxX - 2 * xgap, maxY - 2 * ygap))
        background2[ygap:maxY - ygap, xgap:maxX - xgap] = img2_resize
        background2[avg_image != 1] = [0, 0, 0]

        buffer.clear()
        return avg_image, background2

    return [], []


def analyzeHistogram(path1, path2):
    handCb = cv.imread(path1)
    faceCb = cv.imread(path2)

    facehist = histogram(faceCb)
    handhist = histogram(handCb)
    handhist[175:] = 0
    facehist[175:] = 0

    plt.stem(facehist)
    plt.stem(handhist, linefmt="--")
    plt.show()

    for i in range(5):
        print(np.argmax(handhist))
        handhist[np.argmax(handhist)] = 0


def FindSkinPixel(img, mask_size=20):
    mask = np.ones((mask_size, mask_size)) / np.power(mask_size, 2)

    convolution = cv.filter2D(img, ddepth=-1, kernel=mask)

    diff = np.sqrt(
        np.power(convolution[:, :, 0] - img[:, :, 0], 2) + np.power(convolution[:, :, 1] - img[:, :, 1], 2) + np.power(
            convolution[:, :, 2] - img[:, :, 2], 2))
    diff_thresholded = diff.copy()
    diff_thresholded[diff_thresholded == 0] = 255
    skin_color_indices = np.unravel_index(diff_thresholded.argmin(), diff_thresholded.shape)

    return skin_color_indices[0], skin_color_indices[1]


def adaptiveThreshold(face_histogram, relative_threshold=0.5):
    face_histogram[np.argmax(face_histogram)] = 0

    total = np.sum(face_histogram)
    sum_threshold = (1 - relative_threshold) * 0.5 * total

    lowerT = 0
    upperT = 0

    while np.sum(face_histogram[:lowerT]) < sum_threshold:
        lowerT += 1

    while np.sum(face_histogram[len(face_histogram) - upperT:len(face_histogram)]) < sum_threshold:
        upperT += 1

    return lowerT, len(face_histogram) - upperT


def my_improved_backprojectionV2(img_reference_ycrcb, img_test_ycrcb, int, alpha=55, threshold=80):
    # 1
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 12))
    h, w = np.shape(disc)
    cX, cY = w // 2, h // 2
    M = cv.getRotationMatrix2D((cX, cY), -45, 1.0)
    rotated_disc = cv.warpAffine(disc, M, (w, h))

    histCrCb = cv.calcHist([img_reference_ycrcb], [1, 2], None, [256, 256], [0, 256, 0, 256])
    histCrCb[128, 128] = 0

    y, cr, cb = cv.split(img_test_ycrcb)

    cv.filter2D(histCrCb, -1, rotated_disc, histCrCb)

    cv.normalize(histCrCb, histCrCb, 0, 255, cv.NORM_MINMAX)

    # 2
    means = Y_means(img_reference_ycrcb, intensities=int)

    means_test = means[cr.ravel(), cb.ravel()]
    means_test = means_test.reshape(img_test_ycrcb.shape[:2])

    # 4
    prob = histCrCb[cr.ravel(), cb.ravel()]
    prob = prob.reshape(img_test_ycrcb.shape[:2])
    prob = np.uint8(prob)

    mask = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_normalized = mask / np.sum(mask)

    # cv.filter2D(prob, -1, mask_normalized, prob)

    # cv.normalize(prob1, prob1, 0, 255, cv.NORM_MINMAX)

    mat = np.abs(y - means_test)
    mat = np.uint8(mat)

    result = np.where((mat < alpha), prob, 0)

    zeros = np.zeros_like(result)
    zeros[result >= threshold] = 1

    return zeros, result


def my_improved_backprojectionV3(img_reference_ycrcb, img_test_ycrcb, threshold=80):
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 12))
    h, w = np.shape(disc)
    cX, cY = w // 2, h // 2
    M = cv.getRotationMatrix2D((cX, cY), -45, 1.0)
    rotated_disc = cv.warpAffine(disc, M, (w, h))

    histCrCb = cv.calcHist([img_reference_ycrcb], [1, 2], None, [256, 256], [0, 256, 0, 256])
    histCrCb[128, 128] = 0

    y, cr, cb = cv.split(img_test_ycrcb)

    cv.filter2D(histCrCb, -1, rotated_disc, histCrCb)

    cv.normalize(histCrCb, histCrCb, 0, 255, cv.NORM_MINMAX)

    prob = histCrCb[cr.ravel(), cb.ravel()]
    prob = prob.reshape(img_test_ycrcb.shape[:2])
    prob = np.uint8(prob)

    zeros = np.zeros_like(prob)
    zeros[prob >= threshold] = 1

    return zeros * 255, prob


def Y_means(img, intensities):
    weights = cv.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    weights[:, 128, 128] = 0

    weighted_intensities = np.multiply(weights, intensities)

    sum_weighted_intensities = np.sum(weighted_intensities, axis=0)
    sum_weights = np.sum(weights, axis=0)

    means = np.divide(sum_weighted_intensities, sum_weights, out=np.zeros_like(sum_weighted_intensities),
                      where=sum_weights != 0)

    return means


def CannyMorph(img_ycrcb):
    img_crcb = img_ycrcb.copy()
    img_crcb[:, :, 0] = 0
    img_crcb_bgr = cv.cvtColor(img_crcb, cv.COLOR_YCR_CB2BGR)
    cv.normalize(img_crcb_bgr, img_crcb_bgr, 0, 255, cv.NORM_MINMAX)
    img_crcb_gray = cv.cvtColor(img_crcb_bgr, cv.COLOR_BGR2GRAY)
    cv.normalize(img_crcb_gray, img_crcb_gray, 0, 255, cv.NORM_MINMAX)
    img_crcb_gray_blur = cv.GaussianBlur(img_crcb_gray, (5, 5), 0)
    img_edge = cv.Canny(img_crcb_gray_blur, 50, 100)
    img_edge_morph = Dilation(img_edge, 3)

    return img_edge_morph



def fourier_descriptors(contour, num):
    # create the zeros matrix - num(rows)...num of contours x num(cols)...num of FD
    fd = np.zeros((1, num))
    # go through all the contours
        # create the complex function from the border
    contour_complex = contour[:, 0, 0] + contour[:, 0, 1] * 1j
    # apply the FT and count the amplitude
    ft_contour = np.abs(np.fft.fft(contour_complex))
        # take Fourier coefficients from the second one and normalize them with the length of the border
    fd = ft_contour[1:num + 1] / (len(ft_contour) ** 2)
    return fd

def zpracovat(pocet_deskriptoru, novy):

    if novy:
        adresaDatasetuHand = "/Users/michalprusek/PycharmProjects/Dataset_final/Dataset/hand/"
        adresaDatasetuFace = "/Users/michalprusek/PycharmProjects/Dataset_final/Dataset/face/"
        i=0
        mat = np.zeros((len(os.listdir(adresaDatasetuHand))-3, pocet_deskriptoru))
        gesta = []

        for filename in glob.glob(adresaDatasetuHand + "*.png"):
            with open(os.path.join(os.getcwd(), filename), 'r'):
                img_hand, img_face = pair_images(adresaDatasetuHand, adresaDatasetuFace, filename)

                contour = process(img_hand, img_face)

                FD = fourier_descriptors(contour, pocet_deskriptoru)
                if len(FD) > pocet_deskriptoru - 1:
                    mat[i, :] = FD
                    gesta = np.append(gesta, (ntpath.split(filename)[1])[:ntpath.split(filename)[1].find('_')])
                    print(gesta[i])
                    i += 1

        saveArray = []
        saveArray.append(mat)
        saveArray.append(gesta)

        np.save("deskriptory.npy",saveArray)
    else:
        saveArray = np.load("deskriptory.npy",allow_pickle=True)
        mat = saveArray[0]
        gesta = saveArray[1]

    return mat, gesta

def zkouska(contour, pocet_deskriptoru, mat, gesta, pocetSousedu):
    distances = []
    kNNGestures = []

    FD = fourier_descriptors(contour, pocet_deskriptoru)


    for i in range(mat.shape[0]):
        distances = np.append(distances, np.linalg.norm(FD - mat[i][:pocet_deskriptoru]))

    for i in range(pocetSousedu):
        argMin = np.argmin(distances)
        kNNGestures.append(gesta[argMin])
        distances[argMin] = np.max(distances)

    b = Counter(kNNGestures)

    return b.most_common(1)

def pair_images(adresaDatasetuHand, adresaDatasetuFace, filename):
    adresaFace = os.path.join(adresaDatasetuFace, filename)
    adresaHand = os.path.join(adresaDatasetuHand, filename)
    imgHand = cv.imread(adresaHand)
    imgFace = cv.imread(adresaFace)

    return imgHand, imgFace


def process(img_hand, img_face):
    hand_ycrcb = cv.cvtColor(img_hand, cv.COLOR_BGR2YCR_CB)

    face_blur = cv.GaussianBlur(img_face, (5, 5), 0)

    face_t1 = whiteThreshold(face_blur, threshold=90)

    face_gray = cv.cvtColor(face_t1, cv.COLOR_BGR2GRAY)

    face_otsu = otsu(face_gray)
    face_ROIotsu = ROIotsu(face_gray, vertical_blocks=5, horizontal_blocks=4)

    face_composed = np.clip(face_otsu + face_ROIotsu, 0, 1)

    face_composed_erosion_eyes = Erosion(face_composed, 2)

    face_otsu_color = img_face.copy()
    face_otsu_color[face_composed_erosion_eyes < 1] = [0, 0, 0]

    face_otsu_ycrcb = cv.cvtColor(face_otsu_color, cv.COLOR_BGR2YCR_CB)

    img_backprojected, prob = my_improved_backprojectionV3(face_otsu_ycrcb, hand_ycrcb, threshold=15)


    contours, hierarchy = cv.findContours(img_backprojected, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contour = []
    if len(contours) > 0:
        contour = max(contours, key=lambda x: cv.contourArea(x), default=0)

    return contour


def zpracovat2(file,pocetDeskriptoru,novy):

    if novy:
        mat = np.zeros((8000, pocetDeskriptoru))
        gesta = []
        saveArray = []
        saveArray.append(mat)
        saveArray.append(gesta)
        print(np.shape(mat))
        print(np.shape(gesta))

        np.save(file,saveArray)
    else:
        saveArray = np.load(file,allow_pickle=True)
        mat = saveArray[0]
        gesta = saveArray[1]
        print(np.shape(mat))
        print(np.shape(gesta))

    return mat, gesta
