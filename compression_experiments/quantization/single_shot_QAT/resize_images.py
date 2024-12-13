from PIL import Image
import os
from tqdm import tqdm
# Path to the folder containing the PNG images
folder_path = 'logynthetic/train'

# Iterate through all files in the folder
for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith('.png'):
        # Open the image file
        img = Image.open(os.path.join(folder_path, filename))
        if img.size[1] > 500:
            print(filename)

            # Resize the image to 256x256
            resized_img = img.resize((256, 256))
            
            # Save the resized image back to the folder
            resized_img.save(os.path.join(folder_path, filename))
print("All PNG images have been resized to 256x256")