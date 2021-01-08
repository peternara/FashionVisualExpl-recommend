import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset.dataset import DataLoader
from recommender.models.BPRMF import BPRMF
from recommender.models.VBPR import VBPR
from recommender.models.ExplVBPR import ExplVBPR
from recommender.models.CompVBPR import CompVBPR
from recommender.models.AttentiveFashion import AttentiveFashion
from config.configs import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run train of the Recommender Model.")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--best_metric', type=str, default='hr')
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
    parser.add_argument('--rec', nargs='?', default="comp_vbpr", help="set recommendation model")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--top_k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=-1, help='number of epochs to store model parameters.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--validation', type=bool, default=True, help='True to use validation set, False otherwise')
    parser.add_argument('--restore_epochs', type=int, default=1,
                        help='Default is 1: The restore epochs (Must be lower than the epochs)')

    # Parameters useful during the visual recs
    parser.add_argument('--activated_components', nargs='+', type=int, default=[1, 0, 0, 0],
                        help='[semantic, color, edges, texture]')
    parser.add_argument('--weight_components', nargs='+', type=float, default=[1.0, .0, .0, .0],
                        help='[semantic, color, edges, texture]')
    parser.add_argument('--cnn_model', nargs='?', default='vgg19', help='Model used for feature extraction.')
    parser.add_argument('--output_layer', nargs='?', default='fc2',
                        help='Output layer for feature extraction.')
    parser.add_argument('--embed_k', type=int, default=128, help='Embedding size.')
    parser.add_argument('--embed_d', type=int, default=20, help='size of low dimensionality for visual features')
    parser.add_argument('--attention_layers', type=list, default=[64, 1], help='attention layers')
    parser.add_argument('--l_w', type=float, default=0, help='embedding regularization')
    parser.add_argument('--l_e', type=float, default=0, help='projection matrix regularization')
    parser.add_argument('--l_b', type=float, default=0, help='bias regularization')
    parser.add_argument('--l_f', type=float, default=0, help='cnn regularization')

    return parser.parse_args()


def train():
    args = parse_args()

    if not os.path.exists(results_dir + f'/{args.dataset}/{args.rec}'):
        os.makedirs(results_dir + f'/{args.dataset}/{args.rec}')
    if not os.path.exists(weight_dir + f'/{args.dataset}/{args.rec}'):
        os.makedirs(weight_dir + f'/{args.dataset}/{args.rec}')

    data = DataLoader(params=args)

    print("TRAINING {0} ON {1}".format(args.rec, args.dataset))
    print("PARAMETERS:")
    for arg in vars(args):
        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
    print("\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    if args.rec == 'bprmf':
        model = BPRMF(data, args)
    elif args.rec == 'vbpr':
        model = VBPR(data, args)
    elif args.rec == 'expl_vbpr':
        model = ExplVBPR(data, args)
    elif args.rec == 'comp_vbpr':
        model = CompVBPR(data, args)
    elif args.rec == 'attentive_fashion':
        model = AttentiveFashion(data, args)
    else:
        raise NotImplementedError('Not implemented or unknown Recommender Model.')
    model.train()


if __name__ == '__main__':
    train()
