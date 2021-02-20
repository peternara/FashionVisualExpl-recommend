import numpy as np
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import pandas as pd

all_files = os.listdir('../data/amazon_baby/original/images/')
all_histograms = np.zeros((len(all_files), 8 * 8 * 8), dtype=np.int32)

for idx, file in enumerate(all_files):
    # edges extraction
    image = cv2.imread('../data/amazon_baby/original/images/' + file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Ie1 = cv2.Canny(gray, 255 / 3, 255)
    f = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    Ie2 = cv2.filter2D(gray, -1, f)
    Ie = Ie1 + Ie2
    Ie_end = np.clip(255 - Ie, a_min=0, a_max=255)

    # color histogram extraction
    contours, hierarchy = cv2.findContours(Ie, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_info = []
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    max_contour = sorted(contour_info, key=lambda cont: cont[2], reverse=True)[0]
    mask = np.copy(image)
    cv2.fillPoly(mask, pts=[max_contour[0]], color=(0, 0, 0))
    image_for_histogram = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_for_histogram = image_for_histogram[(mask == 0).all(axis=2)]
    # np.expand_dims((mask == 0).all(axis=2), -1).repeat(3, axis=-1)
    # image_for_histogram[np.expand_dims((mask != 0).all(axis=2), -1).repeat(3, axis=-1)] = -1
    temp = (mask == 0).all(axis=2).astype(np.uint8)
    hist = cv2.calcHist([image_for_histogram], [0, 1, 2], temp, [8, 8, 8], [0, 255, 0, 255, 0, 255])
    hist = np.asarray(hist, dtype=np.int32).flatten()
    all_histograms[idx] = hist

# classes
df = pd.read_csv('../data/amazon_baby/original/classes_vgg19.csv')
count_class = df.groupby('ClassStr').size().reset_index(name='counts')
count_class = count_class.sort_values(by='counts', ascending=False)
print('There are %d different classes' % len(count_class))
one_hot_encoding = LabelBinarizer().fit_transform(df.ClassStr)

np.save('../data/amazon_baby/original/features/histograms.npy', all_histograms)
np.save('../data/amazon_baby/original/features/one_hot_enc.npy', one_hot_encoding)
