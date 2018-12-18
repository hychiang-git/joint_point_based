import os
import sys
import time
import glob
import numpy as np
from PIL import Image
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils import utils 

def readClassesWeights(filename, num_classes):
    if os.path.exists(filename):
        counts = list()
        with open(filename, 'r') as f:
            for index, line in enumerate(f):
                val = line.split()[-1]
                counts.append(float(val))
        counts = np.array(counts)
    else:
        counts = np.ones(num_classes)
        counts[0] = counts[-1] = 0
    return counts

def get_list(scene_dir, color_dir, pixcoord_dir=None, pixmeta_dir=None, label_dir=None, \
             label_ext='png', color_ext='jpg', pixcoord_ext='npz', pixmeta_ext='npz'):
    color_dir = os.path.join(scene_dir, color_dir)
    color_imgs = glob.glob(os.path.join(color_dir, '*.'+color_ext))
    color_imgs = utils.human_sort(color_imgs)
    label_imgs = [None] * len(color_imgs)
    pixcoord_imgs = [None] * len(color_imgs)
    pixmeta_imgs = [None] * len(color_imgs)
    if label_dir is not None:
        label_dir = os.path.join(scene_dir, label_dir)
        label_imgs = [l for l in glob.glob(os.path.join(label_dir, '*.'+label_ext)) if 'vis' not in l]
        label_imgs = utils.human_sort(label_imgs)
        assert len(label_imgs) == len(color_imgs),\
           '[Error] {}, Image length not equal: label {}, color {}' \
           .format(scene_dir, len(label_imgs), len(color_imgs))
    if pixcoord_dir is not None:
        pixcoord_dir = os.path.join(scene_dir, pixcoord_dir)
        pixcoord_imgs = glob.glob(os.path.join(pixcoord_dir, '*.'+pixcoord_ext))
        pixcoord_imgs = utils.human_sort(pixcoord_imgs)
        assert len(pixcoord_imgs) == len(color_imgs),\
           '[Error] {}, Image length not equal: pixcoord {}, color {}' \
           .format(scene_dir, len(pixcoord_imgs), len(color_imgs))
    if pixmeta_dir is not None:
        pixmeta_dir = os.path.join(scene_dir, pixmeta_dir)
        pixmeta_imgs = glob.glob(os.path.join(pixmeta_dir, '*.'+pixmeta_ext))
        pixmeta_imgs = utils.human_sort(pixmeta_imgs)
        assert len(pixmeta_imgs) == len(color_imgs),\
           '[Error] {}, Image length not equal: pixmeta {}, color {}' \
           .format(scene_dir, len(pixmeta_imgs), len(color_imgs))

    return label_imgs, color_imgs, pixcoord_imgs, pixmeta_imgs

def get_image_sample(labelweights, image_path, label_path):
    # Get a random image, label and camera pose from scene
    color_img = Image.open(image_path)  # load color
    color_img = np.array(color_img)
    label_img = Image.open(label_path)  # load label
    label_img = np.array(label_img)
    label_img = np.expand_dims(label_img, axis=-1)
    assert color_img.shape[0:2] == label_img.shape[0:2],\
           'height, width not equal corlor {}, label {}'\
           .format(color_img.shape, label_img.shape)
    height = color_img.shape[0]
    width = color_img.shape[1]

    pixel_weight = labelweights[label_img.reshape(-1)]  # get pixel weight
    pixel_weight = pixel_weight.reshape(height, width, 1)
    occ_image_label_set = set(label_img.reshape(-1)) 
    occ_image_label_set = occ_image_label_set - set({0,1,2})
    return color_img, label_img, pixel_weight, occ_image_label_set

def get_scene_data(labelweights, get_pixmeta, data_dir):
    label_dir, color_dir, pixcoord_dir, pixmeta_dir = data_dir
    color_img, pixel_label, pixel_weight, label_set = \
        get_image_sample(labelweights, color_dir, label_dir)
    if get_pixmeta is True:
        pixel_world_coord = np.load(pixcoord_dir)['pixcoord']
        pixel_meta = np.load(pixmeta_dir)['meta']
        pixel_meta = np.concatenate([pixel_world_coord, pixel_meta], axis=-1)
    else:
        pixel_meta = np.zeros((color_img.shape[0], color_img.shape[1], 15))

    return color_img, pixel_label, pixel_weight, pixel_meta


def get_test_scene_data(get_pixmeta, data_dir):
    color_dir, pixcoord_dir, pixmeta_dir = data_dir
    color_img = np.array(Image.open(color_dir))  # load color
    if get_pixmeta is True:
        pixel_world_coord = np.load(pixcoord_dir)['pixcoord']
        pixel_meta = np.load(pixmeta_dir)['meta']
        pixel_meta = np.concatenate([pixel_world_coord, pixel_meta], axis=-1)
    else:
        pixel_meta = np.zeros((color_img.shape[0], color_img.shape[1], 15))

    return color_img, pixel_meta

