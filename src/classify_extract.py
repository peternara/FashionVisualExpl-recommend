import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from vision.LowFeatureExtractor import *
from vision.CnnFeatureExtractor import *
from vision.Dataset import *
from config.configs import *
from utils.write import *
from utils.read import *

from skimage import io
import numpy as np
import argparse
import time
import sys
import csv
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run experiments')
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset path')
    parser.add_argument('--model_name', nargs='?', default='VGG19', help='model for feature extraction')
    parser.add_argument('--num_colors', type=int, default=3, help='number of dominant colors to extract')
    parser.add_argument('--cnn_output_name', nargs='?', default='block5_conv4', help='output layer name')
    parser.add_argument('--print_each', type=int, default=100, help='print each n samples')

    return parser.parse_args()


def classify_extract():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    if not os.path.exists(colors_path.format(args.dataset)):
        os.makedirs(colors_path.format(args.dataset))
    if not os.path.exists(edges_path.format(args.dataset)):
        os.makedirs(edges_path.format(args.dataset))
    if not os.path.exists(cnn_features_path_split.format(args.dataset,
                                                         args.model_name.lower(),
                                                         args.cnn_output_name)):
        os.makedirs(cnn_features_path_split.format(args.dataset,
                                                   args.model_name.lower(),
                                                   args.cnn_output_name))

    # model setting
    cnn_model = CnnFeatureExtractor(
        model_name=args.model_name,
        output_layer=args.cnn_output_name,
        imagenet=read_imagenet_classes_txt(imagenet_classes_path)
    )

    # low_level_model = LowFeatureExtractor(
    #     num_colors=args.num_colors
    # )

    # dataset setting
    data = Dataset(
        dataset=args.dataset,
        resize=(224, 224),
        model_name=args.model_name
    )
    print('Loaded dataset from %s' % images_path.format(args.dataset))

    # high-level visual features
    # cnn_features_shape = [data.num_samples, *cnn_model.model.get_layer(args.cnn_output_name).output.shape[1:]]
    # cnn_features = np.empty(shape=cnn_features_shape)

    # low-level visual features
    # colors = np.empty(shape=[data.num_samples, args.num_colors * 3])
    # edges = np.empty(shape=cnn_features_shape)

    # classification and features extraction
    print('Starting classification...\n')
    start = time.time()

    with open(classes_path.format(args.dataset, args.model_name.lower()), 'w') as f:
        fieldnames = ['ImageID', 'ClassStr', 'ClassNum', 'Prob']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(data):
            norm_image, original_image, path = d

            # high-level visual features extraction
            out_class = cnn_model.classify(sample=(norm_image, path))
            cnn_features = cnn_model.extract_feature(sample=(norm_image, path))
            save_np(npy=cnn_features,
                    filename=cnn_features_path_split.format(args.dataset,
                                                            args.model_name.lower(),
                                                            args.cnn_output_name) + str(i) + '.npy')
            writer.writerow(out_class)

            # low-level visual feature extraction
            # edge, color, color_image = low_level_model.extract_color_edges(sample=(original_image, path))
            # colors[i] = color
            # edges[i] = cnn_model.extract_feature(
            #     sample=(np.expand_dims(data.resize_and_normalize(Image.fromarray(
            #         cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))), axis=0), path)
            # )
            # io.imsave(edges_path.format(args.dataset) + str(i) + '.tiff', edge)
            # io.imsave(colors_path.format(args.dataset) + str(i) + '.jpg', color_image)

            if (i + 1) % args.print_each == 0:
                sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
                sys.stdout.flush()

    # save_np(npy=cnn_features,
    #         filename=cnn_features_path.format(args.dataset, args.model_name.lower(), args.cnn_output_name))
    # save_np(npy=colors, filename=color_features_path.format(args.dataset))
    # save_np(npy=edges,
    #         filename=edge_features_path.format(args.dataset, args.model_name.lower(), args.cnn_output_name))

    end = time.time()

    print('\n\nClassification and feature extraction completed in %f seconds.' % (end - start))
    print('Saved cnn features numpy to ==> %s' %
          cnn_features_path.format(args.dataset, args.model_name.lower(), args.cnn_output_name))
    print('Saved classification file to ==> %s' % classes_path.format(args.dataset, args.model_name.lower()))
    print('Saved colors features numpy to ==> %s' % color_features_path.format(args.dataset))
    print('Saved edges features numpy to ==> %s' % edge_features_path.format(args.dataset,
                                                                             args.model_name.lower(),
                                                                             args.cnn_output_name))


if __name__ == '__main__':
    classify_extract()
