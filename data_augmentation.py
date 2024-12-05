import albumentations as A
import cv2
import glob
import os
import math
import random

transform = A.Compose([A.Resize(height=512, width=512),
                       A.RandomScale(scale_limit=(-0.5, 0.1), p=0.9),
                       A.RandomCrop(width=256, height=256),
                       A.HorizontalFlip(p=0.5), 
                       A.Rotate(limit=(-45, 45), p=0.6),
                       A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20), p=0.8)],
                       )

resize_original = A.Compose([A.Resize(width=256, height=256)])

output_dir = 'augmented/'
images_per_category = 2000

if not os.path.exists('dataset'):
    print("Dataset not found! Place dataset folder (and name it dataset) and this file in the same directory!")

for folder_path in glob.glob('dataset/*'):
    print("Reading " + folder_path)
    images = glob.glob(folder_path + '/*.jpg')
    total_images = len(images)
    print("images found: " + str(total_images))
    for image_path in images:
        image = cv2.imread(image_path)
        if image is None:
            # For some reason, a few images keep giving an error. Gives error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
            print("Error reading " + image_path)
            continue 

        # Resize original image to standard dimensions. Create directories if they don't exist.
        original_resized_image = resize_original(image=image)['image']
        path = image_path.split(sep='\\')
        category_dir = output_dir + path[1]
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        cv2.imwrite(category_dir + '/' + path[2][:-4] + '_original.jpg', original_resized_image)
        
        # Create augmented images and save to output directory
        images_to_create = math.ceil((images_per_category - total_images) / total_images)
        transformed_images = {}
        for i in range(0, images_to_create):
            transformed_images[i] = transform(image=image)['image']
            cv2.imwrite(category_dir + '/' + path[2][:-4] + '_aug_' + str(i+1) + '.jpg', transformed_images[i])

        # Randomly remove extra images to balance dataset
        while len(glob.glob(category_dir + '/*.jpg')) > images_per_category:
            image_pool = glob.glob(category_dir + '/*_aug_*.jpg')
            remove_image = random.choice(image_pool)
            os.remove(remove_image)