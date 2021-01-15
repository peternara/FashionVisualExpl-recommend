import argparse
import pandas as pd
from config.configs import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run logs to excel.")
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
    parser.add_argument('--rec', nargs='?', default='grad_fashion', help='recommender model')
    parser.add_argument('--file', nargs='?',
                        default='recs-200-batch_256-D_20-K_128-lr_0.005-reg_0.001.tsv',
                        help='tsv file name')

    return parser.parse_args()


def get_explanations():
    args = parse_args()
    users_items_id = pd.read_csv(
        results_dir + '/' + args.dataset + '/' + args.rec + '/' + args.file,
        sep='\t', names=['USER_ID', 'ITEM_ID', 'COLOR', 'EDGES'])
    all_reviews = pd.read_csv(
        all_final.format(args.dataset), sep='\t'
    )
    users_items_reviews = pd.merge(users_items_id, all_reviews, on=['USER_ID', 'ITEM_ID'], how='inner')
    users_items_reviews.drop(['USER', 'ASIN', 'TIME', 'CATEGORY'], axis=1, inplace=True)
    users_items_reviews['DIFF'] = users_items_reviews['COLOR'] - users_items_reviews['EDGES']
    users_items_reviews.sort_values(by='DIFF', ascending=False, inplace=True)
    users_items_reviews.head(50).to_csv(
        results_dir + '/' + args.dataset + '/' + args.rec + '/' + 'color_reviews.tsv',
        sep='\t', index=False
    )
    users_items_reviews.sort_values(by='DIFF', ascending=True, inplace=True)
    users_items_reviews.head(50).to_csv(
        results_dir + '/' + args.dataset + '/' + args.rec + '/' + 'edges_reviews.tsv',
        sep='\t', index=False
    )


if __name__ == '__main__':
    get_explanations()
