import pandas as pd
import argparse
from config.configs import *

parser = argparse.ArgumentParser(description="Run create users-items.")
parser.add_argument('--dataset', nargs='?', default='amazon_women', help='dataset name')

args = parser.parse_args()

interactions = pd.read_csv(all_interactions.format(args.dataset), sep='\t', header=None)
interactions.columns = ['USER', 'ITEM', 'TIME']

count_user = interactions.groupby('USER').size().reset_index(name='counts')
count_user = count_user.sort_values(by='counts', ascending=False)

print(f'''Lowest number of positive items per user: {count_user.iloc[-1, 1]}''')
print(f'''Users with 3 interactions: {len(count_user[count_user['counts'] == 3])}/{len(count_user)}''')
print(f'''Users with 4 interactions: {len(count_user[count_user['counts'] == 4])}/{len(count_user)}''')
print(f'''Users with >=5 interactions: {len(count_user[count_user['counts'] >= 5])}/{len(count_user)}''')
