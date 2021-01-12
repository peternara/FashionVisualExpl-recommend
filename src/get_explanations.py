import argparse
import pandas as pd
from config.configs import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run logs to excel.")
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
    parser.add_argument('--rec', nargs='?', default='grad_fashion', help='recommender model')
    parser.add_argument('--file', nargs='?',
                        default='best-recs-180-batch_64-D_20-K_64-lr_0.001-reg_0.0-attlayers_[64, 1].tsv',
                        help='tsv file name')

    return parser.parse_args()


def get_explanations():
    args = parse_args()
    users_items_id = pd.read_csv(
        results_dir.format(args.dataset) + '/' + args.dataset + '/' + args.rec + '/' + args.file,
        sep='\t', names=['USER', 'ITEM', 'COLOR', 'EDGES'])
    print(1)


if __name__ == '__main__':
    get_explanations()
