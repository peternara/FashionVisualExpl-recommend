# DATASET
data_path = '../data/{0}/'
imagenet_classes_path = '../data/imagenet_classes.txt'
all_items = data_path + 'all_items.csv'
all_interactions = data_path + 'all_interactions.tsv'
all_final = data_path + 'all_final.tsv'
users = data_path + 'users.tsv'
items = data_path + 'items.tsv'
training_path = data_path + 'trainingset.tsv'
validation_path = data_path + 'validationset.tsv'
test_path = data_path + 'testset.tsv'
original = data_path + 'original/'
images_path = original + 'images/'
dataset_info = data_path + 'stats_after_downloading'

classes_path = original + 'classes_{1}.csv'
cnn_features_path = original + 'cnn_features_{1}_{2}.npy'
cnn_features_path_split = original + 'features/cnn_{1}_{2}/'
color_features_path = original + 'color_features.npy'
edge_features_path = original + 'edge_features_{1}_{2}.npy'
texture_features_path = original + 'texture_features_{1}.npy'

features_path = original + 'features/'
hist_color_features_path = features_path + 'histograms.npy'
hist_color_features_path_dir = features_path + 'color_histograms/'
class_features_path = features_path + 'one_hot_enc.npy'
class_features_path_dir = features_path + 'one_hot_encodings/'
colors_path = features_path + 'colors/'
edges_path = features_path + 'edges/'

# RESULTS
weight_dir = '../results/rec_model_weights'
results_dir = '../results/rec_results'
