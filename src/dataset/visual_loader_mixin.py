import numpy as np
from config.configs import *


class VisualLoader:

    def __init__(self, data):
        self.data = data

        self.cnn_features = None
        self.dim_cnn_features = None

        self.color_features = None
        self.dim_color_features = None

        self.edge_features = None
        self.dim_edge_features = None

    def process_cnn_visual_features(self):
        self.cnn_features = \
            np.load(cnn_features_path.format(
                self.data.params.dataset,
                self.data.params.cnn_model,
                self.data.params.output_layer
            )
            )
        self.cnn_features = self.cnn_features / np.max(np.abs(self.cnn_features))
        self.dim_cnn_features = self.cnn_features.shape[1]

    def process_color_visual_features(self):
        self.color_features = np.load(color_features_path.format(self.data.params.dataset))
        self.color_features = self.color_features / np.max(np.abs(self.color_features))
        self.dim_color_features = self.color_features.shape[1]

    def process_edge_visual_features(self):
        self.edge_features = \
            np.load(edge_features_path.format(
                self.data.params.dataset,
                self.data.params.cnn_model,
                self.data.params.output_layer
            )
            )
        self.edge_features = self.edge_features / np.max(np.abs(self.edge_features))
        self.dim_edge_features = self.edge_features.shape[1]
