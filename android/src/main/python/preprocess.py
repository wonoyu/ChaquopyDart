import cv2
import numpy as np
import scipy.fftpack

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def draw_segmentation_mask(w, h, contours, selected):
    bin_cnt = np.zeros((h, w), np.uint8)
    for i, cnt in enumerate(contours):
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if i in selected:
            rxf = rx + rw - 1
            ryf = ry + rh - 1
            cv2.rectangle(bin_cnt, (rx, ry), (rxf, ryf), 255, -1)
    return bin_cnt

def process_mask(gray, rectangles_mask, min_contour_area=20):
    h, w = gray.shape
    final = np.zeros((h, w), np.uint8)
    final = cv2.bitwise_not(final)
    contours, hierarchy = cv2.findContours(rectangles_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1

        _, character = cv2.threshold(gray[ry:ryf, rx:rxf], 127, 255, cv2.THRESH_OTSU)
        # print("character")
        # _, character_contours, _ = cv2.findContours(character, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # for _, character_cnt in enumerate(character_contours):
        #     contour_area = cv2.contourArea(character_cnt)
        #     print(contour_area)
        #     if contour_area > min_contour_area:
        #         final[ry:ryf, rx:rxf] = character
        final[ry:ryf, rx:rxf] = character
    return final

def clean_contours(img_bin, border_radius=10, min_iou_ratio=0.005, max_iou_ratio=0.75):
    h, w = img_bin.shape

    # add a "border" so a char thats in the borders is not detected like a outer contour
    img_bin[0, :] = 255
    img_bin[h - 1, :] = 255
    img_bin[:, 0] = 255
    img_bin[:, w - 1] = 255

    contours, hierarchy = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    maxrect = [0, 0, w, h]
    selected_contours = []
    for i, cnt in enumerate(contours):
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1
        iou_ratio = bb_intersection_over_union(maxrect, [rx, ry, rxf, ryf])
        has_right_size = min_iou_ratio < iou_ratio and iou_ratio < max_iou_ratio
        close_to_border = rxf < border_radius or ryf < border_radius or rx + border_radius > w or ry + border_radius > h
        if has_right_size and not close_to_border:
            selected_contours.append(i)
    return contours, selected_contours

def clean_img_bin(img_bin, min_area, max_area, border_radius=10):
    h, w = img_bin.shape

    # add a "border" so a char thats in the borders is not detected like a outer contour
    img_bin[0, :] = 255
    img_bin[h - 1, :] = 255
    img_bin[:, 0] = 255
    img_bin[:, w - 1] = 255

    character_contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    for i, character_cnt in enumerate(character_contours):
        rx, ry, rw, rh = cv2.boundingRect(character_cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1
        contour_area = cv2.contourArea(character_cnt)
        has_right_area = min_area < contour_area < max_area
        close_to_border = rxf < border_radius or ryf < border_radius or rx + border_radius > w or ry + border_radius > h
        if has_right_area and not close_to_border:
            cv2.drawContours(mask, character_contours, i, (255, 255, 255), cv2.FILLED, 8)

    final2 = cv2.bitwise_and(img_bin, img_bin, mask=mask)
    bk = cv2.bitwise_not(np.zeros((h, w), np.uint8))
    final2_bk = cv2.bitwise_and(bk, bk, mask=cv2.bitwise_not(mask))
    return cv2.bitwise_or(final2, final2_bk)

def homomorphic(gray, low_gamma=0.3, high_gamma=1.5):
    img = gray
    rows, cols = gray.shape

    # based on this excellent answer: http://stackoverflow.com/a/24732434/2692914
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # divides la imagen en alta frecuencia y baja frecuencia
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # las unes con diferentes coeficientes
    gamma1 = low_gamma
    gamma2 = high_gamma
    Ioutlow_gamma = gamma1 * Ioutlow[0:rows, 0:cols]
    Iouthigh_gamma = gamma2 * Iouthigh[0:rows, 0:cols]
    Iout = Ioutlow_gamma + Iouthigh_gamma

    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    return np.array(255 * Ihmf, dtype="uint8")

def preprocess(img):
    image = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    wsize = h >> 3
    gray = cv2.bilateralFilter(gray, wsize, 30, wsize)

    filtered = homomorphic(gray, 0.1, 1.)

    _, img_bin = cv2.threshold(filtered, 127, 255, cv2.THRESH_OTSU)

    contours, selected = clean_contours(img_bin)
    mask = draw_segmentation_mask(w, h, contours, selected)

    final = process_mask(filtered, mask)

    final2 = clean_img_bin(final, 20, h*w*0.5)

    _, decoded = cv2.imencode(".jpg", final2)

    return decoded.tobytes()
