import os
import argparse
import shutil
from config.configs import *

parser = argparse.ArgumentParser(description="Run create users-items.")
parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
parser.add_argument('--final', type=bool, default=True, help='before or after corrupted images deleting')

args = parser.parse_args()

with open(original.format(args.dataset) + ('duplicates' if not args.final else 'duplicates_final'), 'r') as f:
    all_lines = f.readlines()

files_to_copy = list()
files_to_copy.append(all_lines[0].split('\n')[0][2:])
take_next = False
for line in all_lines[1:]:
    if take_next:
        files_to_copy.append(line.split('\n')[0][2:])
    if line == '\n':
        take_next = True
    else:
        take_next = False

with open(original.format(args.dataset) + ('first_of_each' if not args.final else 'first_of_each_final'), 'w') as f:
    for file_to_copy in files_to_copy:
        f.write(file_to_copy + '\n')

os.makedirs(original.format(args.dataset) + ('duplicates_dir/' if not args.final else 'duplicates_dir_final/'), exist_ok=True)
for file in files_to_copy:
    shutil.copy(images_path.format(args.dataset) + file, original.format(args.dataset) + ('duplicates_dir/' if not args.final else 'duplicates_dir_final/') + file)
