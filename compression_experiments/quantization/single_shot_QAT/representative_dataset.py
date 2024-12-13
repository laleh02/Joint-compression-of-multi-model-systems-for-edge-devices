import os
import cv2
import numpy as np
from tqdm import tqdm
folder_path = 'logynthetic/test'
image_files = os.listdir(folder_path)
gt_images = sorted([img for img in image_files if '0.' in img])
lq_images = sorted([img for img in image_files if '0_noisy.' in img])

assert all(["".join(gt.split("_")[0]) == "".join(lq.split("_")[0]) \
                    for (gt, lq) in zip(gt_images, lq_images)])


print(gt_images[0:5])
print(lq_images[0:5])
gt_image_array = []
lq_image_array = []
for img_file in tqdm(gt_images):
    if '0.' in img_file:
        img = cv2.imread(os.path.join(folder_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        gt_image_array.append(img)
for img_file in tqdm(lq_images):
    if '0_noisy.' in img_file:
        img = cv2.imread(os.path.join(folder_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        lq_image_array.append(img)
lq_image_array = np.array(lq_image_array).astype(np.uint8)
lq_image_array = np.transpose(lq_image_array, (0, 3, 1, 2))  # Reshape to [N, 3, 256, 256]

gt_image_array = np.array(gt_image_array).astype(np.uint8)
gt_image_array = np.transpose(gt_image_array, (0, 3, 1, 2))  # Reshape to [N, 3, 256, 256]

np.save('test_lq_dataset_noisy.npy', lq_image_array)  # Save the array to a .npy file
np.save('test_gt_dataset.npy', gt_image_array)  # Save the array to a .npy file