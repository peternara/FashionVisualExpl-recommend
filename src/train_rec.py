import argparse
import os

from dataset.dataset import DataLoader
from recommender.models.VBPR import VBPR
from config.configs import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run train of the Recommender Model.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_fashion', help='dataset name')
    parser.add_argument('--rec', nargs='?', default="acf", help="set recommendation model")
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--k', type=int, default=50, help='top-k of recommendation.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=50, help='number of epochs to store model parameters.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--validation', type=int, default=1, help='1 to use validation set, 0 otherwise')
    parser.add_argument('--restore_epochs', type=int, default=1,
                        help='Default is 1: The restore epochs (Must be lower than the epochs)')

    # Parameters useful during the visual recs
    parser.add_argument('--embed_k', type=int, default=64, help='Embedding size.')
    parser.add_argument('--embed_d', type=int, default=20, help='size of low dimensionality')
    parser.add_argument('--lambda1', type=float, default=1.0, help='lambda1 DVBPR')
    parser.add_argument('--lambda2', type=float, default=0.001, help='lambda2 DVBPR')
    parser.add_argument('--layers_component', type=list, default=[64, 1], help='list component level layers for ACF')
    parser.add_argument('--layers_item', type=list, default=[64, 1], help='list item level layers for ACF')
    parser.add_argument('--l_w', type=float, default=0.01, help='size of low dimensionality')
    parser.add_argument('--l_b', type=float, default=1e-2, help='size of low dimensionality')
    parser.add_argument('--l_e', type=float, default=0, help='size of low dimensionality')

    return parser.parse_args()


def train():
    args = parse_args()

    if not os.path.exists(results_dir + f'/{args.dataset}'):
        os.makedirs(results_dir + f'/{args.dataset}')
    if not os.path.exists(weight_dir + f'/{args.dataset}'):
        os.makedirs(weight_dir + f'/{args.dataset}')

    data = DataLoader(params=args)

    print("TRAINING {0} ON {1}".format(args.rec, args.dataset))
    print("PARAMETERS:")
    for arg in vars(args):
        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
    print("\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.rec == 'vbpr':
        model = VBPR(data, args)
    else:
        raise NotImplementedError('Not implemented or unknown Recommender Model.')
    model.train()


if __name__ == '__main__':
    train()
