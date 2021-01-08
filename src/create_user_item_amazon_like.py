import os
import pandas as pd
import argparse
import shutil
import sys
from config.configs import *

parser = argparse.ArgumentParser(description="Run create users-items.")
parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')

args = parser.parse_args()

available_images = os.listdir(images_path.format(args.dataset))
available_images = [file.split('.')[0] for file in available_images]
available = pd.DataFrame(available_images, columns=['ASIN'])

df_users_items = pd.read_csv(data_path.format(args.dataset) + 'all.tsv', sep='\t')
available_interactions = pd.merge(df_users_items, available, on='ASIN', how='inner')

available_interactions['USER_ID'] = available_interactions.groupby('USER').grouper.group_info[0]
available_interactions['ITEM_ID'] = available_interactions.groupby('ASIN').grouper.group_info[0]
available_interactions = available_interactions.sort_values(by='USER_ID')

# check k-core
count_user = available_interactions.groupby('USER_ID').size().reset_index(name='counts')
count_user = count_user.sort_values(by='counts', ascending=False)

print('Statistics (after downloading images):')
print(f'''Lowest number of positive items per user: {count_user.iloc[-1, 1]}''')
print(f'''Users: {len(available_interactions['USER'].unique())}''')
print(f'''Items: {len(available_interactions['ASIN'].unique())}''')
print(f'''Interactions: {len(available_interactions)}''')
print(f'''Sparsity: {1 - (len(available_interactions) / (len(available_interactions['USER'].unique()) * 
                                                  len(available_interactions['ASIN'].unique())))}''')

available_interactions.to_csv(data_path.format(args.dataset) + 'all_final.tsv', index=False, sep='\t')

users_items = pd.concat([available_interactions['USER_ID'],
                         available_interactions['ITEM_ID'],
                         available_interactions['TIME']], axis=1)
users_items.to_csv(all_interactions.format(args.dataset), index=False, header=False, sep='\t')

available_interactions = available_interactions.reset_index(drop=True)
users_df = pd.concat([pd.Series(available_interactions['USER'].unique()),
                      pd.Series(available_interactions['USER_ID'].unique())], axis=1)
items_df = pd.concat([pd.Series(available_interactions['ASIN'].unique()),
                      pd.Series(available_interactions['ITEM_ID'].unique())], axis=1).sort_values(by=1)
users_df.to_csv(users.format(args.dataset), index=False, header=False, sep='\t')
items_df.to_csv(items.format(args.dataset), index=False, header=False, sep='\t')

for i, row in items_df.iterrows():
    source_file = images_path.format(args.dataset) + str(row[0]) + '.jpg'
    dest_file = images_path.format(args.dataset) + str(row[1]) + '.jpg'
    shutil.move(source_file, dest_file)
    sys.stdout.write('\r%d/%d samples completed' % (i + 1, len(items_df)))
    sys.stdout.flush()
