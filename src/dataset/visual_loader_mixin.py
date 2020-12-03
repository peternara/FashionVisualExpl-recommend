import numpy as np
from config.configs import *


class VisualLoader:

    def __init__(self, data):
        self.data = data

        self.visual_features = None
        self.dim_visual_feature = None

        self.color_features = None
        self.dim_color_features = None

        self.texture_features = None
        self.dim_texture_features = None

    def process_visual_features(self):
        self.visual_features = \
            np.load(features_path.format(
                self.data.params.dataset,
                self.data.params.cnn_model,
                self.data.params.output_layer
            )
            )
        self.visual_features = self.visual_features / np.max(np.abs(self.visual_features))
        self.dim_visual_feature = self.visual_features.shape[1]

    def process_expl_visual_features(self):
        self.color_features = np.load(color_features_path.format(self.data.params.dataset))
        self.color_features = self.color_features / np.max(np.abs(self.color_features))
        self.dim_color_features = self.color_features.shape[1]

        self.texture_features = np.load(texture_features_path.format(
            self.data.params.dataset,
            self.data.params.cnn_model
        )
        )
        self.texture_features = self.texture_features / np.max(np.abs(self.texture_features))
        self.dim_texture_features = self.texture_features.shape[1]
