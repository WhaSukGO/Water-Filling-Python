import cv2
import numpy as np


def minMaxLoc(val):
    return 0 if val > 0 else val


def minMaxLoc(img):
    pass


def water_filling(img):

    h, w = img.shape

    neta = 0.2

    # Water
    w_ = np.zeros((h, w), dtype=np.float)

    # Overall height
    G_ = np.zeros((h, w), dtype=np.float)

    h_ = img.copy()

    for t in range(100):
        G_ = w_ + h_

        # Find peak
        G_peak = np.amax(G_)

        pouring = np.exp(-t) * (G_peak - G_)

        x = np.linspace(1, w-2, w-2)
        y = np.linspace(1, h-2, h-2)

        X, Y = np.meshgrid(x, y)
        X = X.astype(np.uint)
        Y = Y.astype(np.uint)

        lx, ly = X-1, Y
        rx, ry = X+1, Y
        tx, ty = X, Y-1
        bx, by = X, Y+1

        left = -G_[Y, X] + G_[ly, lx]
        left[left > 0] = 0

        right = -G_[Y, X] + G_[ry, rx]
        right[right > 0] = 0

        top = -G_[Y, X] + G_[ty, tx]
        top[top > 0] = 0

        btm = -G_[Y, X] + G_[by, bx]
        btm[btm > 0] = 0

        del_w = neta * (left + right + top + btm)

        # del_w : (w-2) * (h-2)
        # pouring : w * h
        # w_ : w * h

        # To match the shape of del_w, padding is required
        del_w = np.pad(del_w, ((1, 1), (1, 1)),
                       'constant', constant_values=0)

        temp = del_w + pouring + w_

        temp[temp < 0] = 0

        w_ = temp

    return G_


def incre_filling(h_, original):

    h_ = h_.astype(np.float)
    original = original.astype(np.float)

    h, w = h_.shape

    neta = 0.2

    # Water
    w_ = np.zeros((h, w), dtype=np.float)

    # Overall height
    G_ = np.zeros((h, w), dtype=np.float)

    for t in range(100):
        G_ = w_ + h_

        # # # Find peak
        # G_peak = np.amax(G_)

        # pouring = np.exp(-t) * (G_peak - G_)

        x = np.linspace(1, w-2, w-2)
        y = np.linspace(1, h-2, h-2)

        X, Y = np.meshgrid(x, y)
        X = X.astype(np.uint)
        Y = Y.astype(np.uint)

        lx, ly = X-1, Y
        rx, ry = X+1, Y
        tx, ty = X, Y-1
        bx, by = X, Y+1

        left = -G_[Y, X] + G_[ly, lx]

        right = -G_[Y, X] + G_[ry, rx]

        top = -G_[Y, X] + G_[ty, tx]

        btm = -G_[Y, X] + G_[by, bx]

        del_w = neta * (left + right + top + btm)

        # del_w : (w-2) * (h-2)
        # pouring : w * h
        # w_ : w * h

        # To match the shape of del_w, padding is required
        del_w = np.pad(del_w, ((1, 1), (1, 1)),
                       'constant', constant_values=0)

        temp = del_w + w_

        temp[temp < 0] = 0

        w_ = temp

        # print(f"[MSG] {t}th iteration")
        # print(np.amax(w_))
        # print(np.amin(w_))

    output = (0.85 * original) / G_ * 255
    output = output.astype(np.uint8)

    return output


def main():
    img = cv2.imread("./original_14_small.png")

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycrcb)

    G_ = water_filling(y)

    G_ = incre_filling(G_, y)
    cv2.imshow("G_", G_)

    # merged = cv2.merge([G_, cr, cb], 3)
    # merged = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    cv2.imshow("y", y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
