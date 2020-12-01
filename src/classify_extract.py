from vision.LowFeatureExtractor import *
from vision.CnnFeatureExtractor import *
from vision.Dataset import *
from config.configs import *
from utils.write import *
from utils.read import *
import pandas as pd
import argparse
import numpy as np
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run experiments')
    parser.add_argument('--dataset', nargs='?', default='amazon_clothing', help='dataset path')
    parser.add_argument('--model_name', nargs='?', default='VGG19', help='model for feature extraction')
    parser.add_argument('--num_colors', type=int, default=3, help='number of dominant colors to extract')
    parser.add_argument('--cnn_output_name', nargs='?', default='fc2', help='output layer name')
    parser.add_argument('--texture_output_layers', type=list, default=[
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ], help='output layers for gram matrix')

    return parser.parse_args()


def classify_extract():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # model setting
    cnn_model = CnnFeatureExtractor(
        model_name=args.model_name,
        output_layer=args.cnn_output_name,
        imagenet=read_imagenet_classes_txt(imagenet_classes_path)
    )

    texture_model = LowFeatureExtractor(
        model_name=args.model_name,
        num_colors=args.num_colors,
        output_layers=args.texture_output_layers
    )

    # dataset setting
    data = Dataset(
        dataset=args.dataset,
        resize=(224, 224),
        model_name=args.model_name
    )
    print('Loaded dataset from %s' % images_path.format(args.dataset))

    # features and classes
    # high-level visual features
    df = pd.DataFrame([], columns={'ImageID', 'ClassStr', 'ClassNum', 'Prob'})
    cnn_features_shape = [data.num_samples, *cnn_model.model.output.shape[1:]]
    cnn_features = np.empty(shape=cnn_features_shape)
    # low-level visual features
    colors = np.empty(shape=[data.num_samples, args.num_colors, 3])
    edges = []
    textures = []

    # classification and features extraction
    print('Starting classification...\n')
    start = time.time()

    for i, d in enumerate(data):
        norm_image, original_image, path = d

        # high-level visual features extraction
        out_class = cnn_model.classify(sample=(norm_image, path))
        cnn_features[i] = cnn_model.model.predict(norm_image, batch_size=1)
        df = df.append(out_class, ignore_index=True)

        # low-level visual feature extraction
        edge, color = texture_model.extract_color_edges(sample=(original_image, path))
        texture = texture_model.extract_texture(sample=(norm_image, path))
        colors[i] = color
        edges.append(edge)
        textures.append(texture)

        if (i + 1) % 100 == 0:
            sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
            sys.stdout.flush()

    write_csv(df=df, filename=classes_path.format(args.dataset, args.model_name.lower()))
    save_np(npy=cnn_features,
            filename=cnn_features_path.format(args.dataset, args.model_name.lower(), args.output_name))
    save_np(npy=colors, filename=color_features_path.format(args.dataset))
    save_obj(obj=edges, name=edge_features_path.format(args.dataset))
    save_obj(obj=textures, name=texture_features_path.format(args.dataset, args.model_name))

    end = time.time()

    print('\n\nClassification and feature extraction completed in %f seconds.' % (end - start))
    print('Saved cnn features numpy to ==> %s' % cnn_features_path.format(args.dataset))
    print('Saved classification file to ==> %s' % classes_path.format(args.dataset))
    print('Saved colors features numpy to ==> %s' % color_features_path.format(args.dataset))
    print('Saved edges features to ==> %s' % edge_features_path.format(args.dataset))
    print('Saved textures features to ==> %s' % texture_features_path.format(args.dataset, args.model_name))


if __name__ == '__main__':
    classify_extract()
