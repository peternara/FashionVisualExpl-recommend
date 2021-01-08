import os
import pandas as pd
import argparse
import shutil
import math
from config.configs import *

parser = argparse.ArgumentParser(description="Run create users-items.")
parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')

args = parser.parse_args()

downloaded_images = os.listdir(images_path.format(args.dataset))  # available images (no duplicates no 404 codes)
downloaded_images = [file.split('.')[0] for file in downloaded_images]
downloaded = pd.DataFrame(downloaded_images, columns=['ASIN'])
downloaded['Downloaded'] = pd.Series([True] * len(downloaded))

all_images = pd.read_csv(all_items.format(args.dataset))  # all items

all_images = pd.merge(all_images, downloaded, on='ASIN', how='outer')
# all items, NaN where we have either duplicates or 404 codes

available_images = all_images.groupby('URL').filter(lambda g: g.isnull().values.sum() < len(g))
# filter out 404 codes from all images

for i, row in available_images.iterrows():
    if math.isnan(row['Downloaded']):
        source_file = images_path.format(args.dataset) + \
                      str(available_images[(available_images['URL'] == row['URL']) &
                                           (available_images['Downloaded'])]['ASIN'].iloc[0]) + '.jpg'
        dest_file = images_path.format(args.dataset) + str(row['ASIN']) + '.jpg'
        shutil.copy(source_file, dest_file)
