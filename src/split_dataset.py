from config.configs import *
import pandas as pd
import argparse
import random

random.seed(1234)

parser = argparse.ArgumentParser(description="Run dataset splitting.")
parser.add_argument('--dataset', nargs='?', default='amazon_clothing', help='dataset name')
parser.add_argument('--validation', type=bool, default=False, help='True --> use validation, False --> no validation')
parser.add_argument('--shuffle', type=bool, default=True, help='dataset shuffle')
parser.add_argument('--column_stratify', type=list, default=[0], help='list of columns to use for stratification')
args = parser.parse_args()

# read all interactions
df = pd.read_csv(all_interactions.format(args.dataset), delimiter='\t', header=None)
df = df.groupby(args.column_stratify).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
df_grouped = df.groupby(args.column_stratify)

if args.validation:
    validation = df_grouped.tail(1).drop_duplicates()
    validation[2] = [1.0] * len(validation)
    validation.to_csv(validation_path.format(args.dataset), index=False, sep='\t', header=None)
    df.drop(index=validation.index, inplace=True)

df_grouped = df.groupby(args.column_stratify)
test = df_grouped.tail(1).drop_duplicates()
train = df.drop(index=test.index)

train[2] = [1.0] * len(train)
test[2] = [1.0] * len(test)

# write to file
train.to_csv(training_path.format(args.dataset), index=False, sep='\t', header=None)
test.to_csv(test_path.format(args.dataset), index=False, sep='\t', header=None)
