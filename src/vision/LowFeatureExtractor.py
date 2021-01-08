from sklearn.cluster import KMeans
import numpy as np
import cv2


def _centroid_histogram(topk):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, topk + 1)
    (hist, _) = np.histogram(np.arange(0, topk), bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def _plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((200, 600, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 600)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 200),
                      color.tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


class LowFeatureExtractor:
    def __init__(self, num_colors):
        self.num_colors = num_colors

    def extract_color_edges(self, sample):
        image, filename = sample

        # edges extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Ie1 = cv2.Canny(gray, 255 / 3, 255)
        f = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Ie2 = cv2.filter2D(gray, -1, f)
        Ie = Ie1 + Ie2
        Ie_end = np.clip(255 - Ie, a_min=0, a_max=255)
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

        # dominant colors extraction
        # image_for_clustering = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) / np.float32(255)
        image_for_clustering = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / np.float32(255)
        image_for_clustering = image_for_clustering[(mask == 0).all(axis=2)]

        clt = KMeans(n_clusters=self.num_colors, random_state=1234)
        clt.fit(image_for_clustering)

        dominant_colors = (clt.cluster_centers_ * 255).astype("uint8")
        # dominant_colors = cv2.cvtColor(np.expand_dims(dominant_colors, axis=0), cv2.COLOR_Lab2RGB)
        dominant_colors = np.expand_dims(dominant_colors, axis=0)
        dominant_colors = dominant_colors[0]

        hist = _centroid_histogram(topk=self.num_colors)
        dominant_colors_image = _plot_colors(hist, dominant_colors)

        return Ie_end, dominant_colors.flatten(), dominant_colors_image
