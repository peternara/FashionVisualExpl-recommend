import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset.dataset import DataLoader
from recommender.models.BPRMF import BPRMF
from recommender.models.VBPR import VBPR
from recommender.models.ACF import ACF
from recommender.models.GradFashion import GradFashion
from recommender.models.AttentiveFashion import AttentiveFashion
from config.configs import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run train of the Recommender Model.")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--best_metric', type=str, default='ndcg')
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
    parser.add_argument('--rec', nargs='?', default="attentive_fashion", help="set recommendation model")
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--top_k', type=int, default=20, help='top-k of recommendation.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=-1, help='number of epochs to store model parameters.')
    parser.add_argument('--batch_eval', type=int, default=128, help='batch size on items for evaluation.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--validation', type=bool, default=True, help='True to use validation set, False otherwise')
    parser.add_argument('--restore_epochs', type=int, default=1,
                        help='Default is 1: The restore epochs (Must be lower than the epochs)')

    # Parameters useful during the visual recs
    parser.add_argument('--list_of_regs', nargs='+', type=float,
                        default=[0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1], help='list of regularization terms')
    parser.add_argument('--layers_component', type=list, default=[64, 1], help='list component level layers for ACF')
    parser.add_argument('--layers_item', type=list, default=[64, 1], help='list item level layers for ACF')
    parser.add_argument('--attention_layers', type=list, default=[64, 1], help='list of attention layers')
    parser.add_argument('--cnn_model', nargs='?', default='vgg19', help='Model used for feature extraction.')
    parser.add_argument('--output_layer', nargs='?', default='fc2',
                        help='Output layer for feature extraction.')
    parser.add_argument('--embed_k', type=int, default=32, help='Embedding size.')
    parser.add_argument('--embed_d', type=int, default=20, help='size of low dimensionality for visual features')
    parser.add_argument('--reg', type=float, default=0, help='regularization')

    return parser.parse_args()


def train():
    args = parse_args()

    if not os.path.exists(results_dir + f'/{args.dataset}/{args.rec}'):
        os.makedirs(results_dir + f'/{args.dataset}/{args.rec}')
    if not os.path.exists(weight_dir + f'/{args.dataset}/{args.rec}'):
        os.makedirs(weight_dir + f'/{args.dataset}/{args.rec}')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    for it, current_reg in enumerate(list(args.list_of_regs)):
        print('--------------------------------------------------------------------')
        print('ITERATION %d/%d WITH REGULARIZATION: %f' % (it + 1, len(list(args.list_of_regs)), current_reg))
        data = DataLoader(params=args)

        print("Training {0} on {1}".format(args.rec, args.dataset))
        print("Parameters:")

        # Setting current reg
        args.reg = current_reg

        for arg in vars(args):
            print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
        print("\n")

        if args.rec == 'bprmf':
            model = BPRMF(data, args)
        elif args.rec == 'vbpr':
            model = VBPR(data, args)
        elif args.rec == 'acf':
            model = ACF(data, args)
        elif args.rec == 'grad_fashion':
            model = GradFashion(data, args)
        elif args.rec == 'attentive_fashion':
            model = AttentiveFashion(data, args)
        else:
            raise NotImplementedError('Not implemented or unknown Recommender Model.')
        model.train()
        print('END REGULARIZATION')
        print('--------------------------------------------------------------------')


if __name__ == '__main__':
    train()
