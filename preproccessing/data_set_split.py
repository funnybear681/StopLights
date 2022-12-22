
import os
import shutil
from PIL import Image
import numpy as np

image_list = []

for root, dirs, files in os.walk(r'./saved'):
    for file in files:
        if file.endswith('.png'):
            image_list.append(os.path.join(root, file))

image_list = np.array(image_list)
np.random.shuffle(image_list)

train_percent = .7
valid_percent = .2
test_percent = .1

total_images = len(image_list)

train_set = np.array([0]*int(total_images*train_percent)+ [1]*int(total_images*valid_percent)+ [2]*int(total_images*test_percent))

while train_set.shape[0] != total_images:
    train_set = np.append(train_set, 0)

np.random.shuffle(train_set)

def convert_to_jpg(image, image_name, save_folder):
    im1 = Image.open(image)
    im1 = im1.convert('RGB')
    im1.save(save_folder+image_name+'.jpg')

for image, code in zip(image_list, train_set):
    image_name = image.split('/')[-1]
    image_name = image_name.split('.')[0]

    if code == 0:
        shutil.copyfile('./saved/'+image_name+'.xml', './data/train/'+image_name+'.xml')
        convert_to_jpg(image, image_name, './data/train/')
    
    if code == 1:
        shutil.copyfile('./saved/'+image_name+'.xml', './data/val/'+image_name+'.xml')
        convert_to_jpg(image, image_name, './data/val/')


    if code == 2:
        shutil.copyfile('./saved/'+image_name+'.xml', './data/test/'+image_name+'.xml')
        convert_to_jpg(image, image_name, './data/test/')