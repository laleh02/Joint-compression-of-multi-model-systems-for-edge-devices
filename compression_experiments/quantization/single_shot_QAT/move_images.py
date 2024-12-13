import os
import shutil
import argparse
from tqdm import tqdm
# Create an argument parser
parser = argparse.ArgumentParser(description='Move and rename files based on certain conditions')

# Add arguments
# parser.add_argument('--partition', help='Path to the index partition file')
parser.add_argument('--source', help='Path to the source folder')
parser.add_argument('--destination', help='Path to the destination folder')

# Parse the arguments
args = parser.parse_args()

# Read the list of numbers from the provided .txt file
# with open(args.partition, 'r') as file:
#     numbers = file.read().split('\n')
#     print(sorted(numbers))
# Iterate over the files in the source folder
for filename in tqdm(os.listdir(args.source)):
    if filename.endswith(".png"):
        parts = filename.split('_')
        A = int(parts[0])
        B = parts[1]
        # Check if B = 0 and A is in the list of numbers
        if '1.png' in B:
            # Move the file and rename it
            new_filename = f"{A}.png"
            shutil.copy(os.path.join(args.source, filename), os.path.join(args.destination, new_filename))