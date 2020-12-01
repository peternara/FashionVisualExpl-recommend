from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
import cv2


class LowFeatureExtractor:
    def __init__(self, model_name, num_colors, output_layers):
        self.model_name = model_name
        self.num_colors = num_colors
        self.output_layers = output_layers

        if self.model_name == 'ResNet50':
            self.model = tf.keras.applications.ResNet50()
        elif self.model_name == 'VGG19':
            self.model = tf.keras.applications.VGG19()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def extract_color_edges(self, sample):
        image, filename = sample

        # edges extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Ie1 = cv2.Canny(gray, 255 / 3, 255)
        f = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Ie2 = cv2.filter2D(gray, -1, f)
        Ie = Ie1 + Ie2
        contours, hierarchy = cv2.findContours(np.clip(Ie, a_min=0, a_max=255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_info = []
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda cont: cont[2], reverse=True)
        max_contour = contour_info[0]
        mask = np.copy(image)
        cv2.fillPoly(mask, pts=[max_contour[0]], color=(0, 0, 0))

        # dominant colors extraction
        image_for_clustering = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2LAB)
        image_for_clustering = image_for_clustering[(mask == 0).all(axis=2)]

        clt = KMeans(n_clusters=self.num_colors, random_state=1234)
        clt.fit(image_for_clustering)

        dominant_colors = clt.cluster_centers_
        return mask, dominant_colors

    def extract_texture(self, sample):
        image, filename = sample
        gram_matrices = []

        for layer in self.output_layers:
            current_model = tf.keras.Model(self.model.input, self.model.get_layer(layer).output)
            current_f_maps = current_model.predict(image, batch_size=1)
            current_f_maps = tf.reshape(current_f_maps,
                                        [current_f_maps.shape[0], current_f_maps.shape[1] * current_f_maps.shape[2]])
            current_gram_matrix = tf.matmul(current_f_maps,
                                            current_f_maps,
                                            transpose_a=True) / np.prod(current_f_maps.shape)
            gram_matrices.append(current_gram_matrix)

        return gram_matrices
