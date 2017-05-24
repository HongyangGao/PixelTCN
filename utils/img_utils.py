import os
import scipy
import scipy.misc
import h5py
import numpy as np


def center_crop(image, pre_height, pre_width, height, width):
    h, w = image.shape[:2]
    j, i = int((h - pre_height)/2.), int((w - pre_width)/2.)
    return scipy.misc.imresize(
        image[j:j+pre_height, i:i+pre_width], [height, width])


def transform(image, pre_height, pre_width, height, width, is_crop):
    if is_crop:
        new_image = center_crop(image, pre_height, pre_width, height, width)
    else:
        new_image = scipy.misc.imresize(image, [height, width])
    return np.array(new_image)/127.5 - 1.


def imread(path, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    return scipy.misc.imread(path).astype(np.float)


def imsave(image, path):
    label_colours = [
        (0,0,0),
        # 0=background
        (128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),
        # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
        (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    images = np.ones(list(image.shape)+[3])
    for j_, j in enumerate(image):
        for k_, k in enumerate(j):
            if k < 21:
                images[j_, k_] = label_colours[int(k)]
    scipy.misc.imsave(path, images)


def get_images(paths, pre_height, pre_width, height, width,
               is_crop=False, is_grayscale=False):
    images = []
    for path in paths:
        image = imread(path, is_grayscale)
        new_image = transform(
            image, pre_height, pre_width, height, width, is_crop)
        images.append(new_image)
    return np.array(images).astype(np.float32)


def save_data(path, image_folder='./images/', label_folder='./labels/'):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    data_file = h5py.File(path, 'r')
    for index in range(data_file['X'].shape[0]):
        scipy.misc.imsave(image_folder+str(index)+'.png', data_file['X'][index])
        imsave(data_file['Y'][index], label_folder+str(index)+'.png')


def compose_images(ids, wides, folders, name):
    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    id_imgs = []
    for i, index in enumerate(ids):
        imgs = []
        for folder in folders:
            path = folder + str(index) +'.png'
            cur_img = scipy.misc.imread(path).astype(np.float)
            cur_img = scipy.misc.imresize(cur_img, [256, int(256*wides[i])])
            imgs.append(cur_img)
            imgs.append(np.ones([3]+list(cur_img.shape)[1:])*255)
        img = np.concatenate(imgs[:-1], axis=0)
        id_imgs.append(img)
        id_imgs.append(np.ones((img.shape[0], 2, img.shape[2]))*255)
    id_img = np.concatenate(id_imgs[:-1], axis=1)
    scipy.misc.imsave(result_folder+name+'.png', id_img)

if __name__ == '__main__':
    folders = ['./images/', './labels/', './samples3/', './samples1/', './samples2/']
    pre_folders = ['./images/', './labels/', './samples3/', './samples2/']
    # folders = ['./images/', './labels/', './samples/']
    ids = [214, 238, 720, 256, 276,277,298,480,571,920,1017,1422]
    wides = [1]*len(ids)
    ids_pre = [15,153,160,534,906]
    pre_wides = [1.3, 1.2, 1.8, 1.1, 1.1]
    compose_images(ids_pre, pre_wides, pre_folders, 'pre_result')
    compose_images(ids, wides, folders, 'result')
