import glob
import os
import sys
import time
import imageio
import numpy as np
import multiprocessing as mp
from functools import partial
from PIL import Image
from collections import Counter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
import dataset_utils


class ScannetDatasetTrain():
    def __init__(self, root, num_classes=21):
        self.root = root
        self.num_classes = num_classes
        self.data_dir = os.path.join(self.root, 'train')
        self.scene_list = glob.glob(os.path.join(self.data_dir, 'scene*'))
        print('[Dataset Train Info] Found ', len(self.scene_list), ' scenes under ', self.data_dir)
        self.labelweights = np.ones(self.num_classes)
        weight_file = os.path.join(self.root, 'class_weights.txt')
        if os.path.exists(weight_file):
            self.labelweights = dataset_utils.readClassesWeights(weight_file, self.num_classes)
        print('[Dataset Train Info] Training labelweights:\n', self.labelweights)

    def __len__(self):
        return len(self.scene_list)


    def __getitem__(self, index):
        scene_dir = self.scene_list[index]
        scene_name = os.path.basename(scene_dir)

        label_imgs, color_imgs, pixcoord_imgs, pixmeta_imgs = \
            dataset_utils.get_list(scene_dir=scene_dir, 
                     label_dir='pseudo_pose_label', color_dir='pseudo_pose_color',\
                     pixcoord_dir='pseudo_pose_pixel_coord', pixmeta_dir='pseudo_pose_pixel_meta',\
                     label_ext='png', color_ext='jpg', pixcoord_ext='npz', pixmeta_ext='npz')

        num_sample = 0
        while num_sample < 10:
            rand_img_idx = np.random.randint(low=0, high=len(label_imgs))
            color_img, pixel_label, pixel_weight, label_set = \
                dataset_utils.get_image_sample(self.labelweights, color_imgs[rand_img_idx], label_imgs[rand_img_idx])
            num_sample += 1
            if len(label_set) >= 2: # valid sample
                break
        return scene_name, color_img, pixel_label, pixel_weight



class ScannetDatasetVal():
    def __init__(self, root, num_classes=21, split='val', \
                 get_pixmeta=False, get_scene_point=False, \
                 frame_skip=1, num_proc=20):
        self.root = root
        self.split = split
        self.get_pixmeta = get_pixmeta
        self.get_scene_point = get_scene_point
        self.num_classes = num_classes
        self.frame_skip = frame_skip
        self.pool = mp.Pool(num_proc)
        self.data_dir = os.path.join(self.root, '%s'%self.split)
        self.scene_list = glob.glob(os.path.join(self.data_dir, 'scene*'))
        print('[Dataset Val Info] Found ', len(self.scene_list), ' scenes under ', self.data_dir)
        self.labelweights = np.ones(self.num_classes)
        weight_file = os.path.join(self.root, 'class_weights.txt')
        if os.path.exists(weight_file):
            self.labelweights = dataset_utils.readClassesWeights(weight_file, self.num_classes)
        print('[Dataset Val Info] Training labelweights:\n', self.labelweights)

    def __len__(self):
        return len(self.scene_list)


    def __getitem__(self, index):
        start_time = time.time()
        scene_dir = self.scene_list[index]
        scene_name = os.path.basename(scene_dir)
        scene_point = None
        if self.get_scene_point is True:
            scene_point = np.load(os.path.join(scene_dir, scene_name+'.npy'))

        label_imgs, color_imgs, pixcoord_imgs, pixmeta_imgs = \
            dataset_utils.get_list(scene_dir=scene_dir, 
                     label_dir='pseudo_pose_label', color_dir='pseudo_pose_color',\
                     pixcoord_dir='pseudo_pose_pixel_coord', pixmeta_dir='pseudo_pose_pixel_meta',\
                     label_ext='png', color_ext='jpg', pixcoord_ext='npz', pixmeta_ext='npz')

        data_dir = zip(label_imgs[::self.frame_skip], color_imgs[::self.frame_skip], \
                       pixcoord_imgs[::self.frame_skip], pixmeta_imgs[::self.frame_skip])
        get_scene_data = partial(dataset_utils.get_scene_data, 
                                 self.labelweights, 
                                 self.get_pixmeta)
        res = self.pool.map(get_scene_data, data_dir)
        
        color_img_list, pixel_label_list, pixel_weight_list, pixel_meta_list = map(list, zip(*res))
        
        print('get {} time {}'.format(scene_name, time.time()-start_time))
        scene_data = {
            'scene_name': scene_name, 
            'scene_point': scene_point, 
            'num_view': len(color_img_list),
            'color_img_list': color_img_list, 
            'pixel_label_list': pixel_label_list, 
            'pixel_weight_list': pixel_weight_list, 
            'pixel_meta_list': pixel_meta_list,
        }    
        return scene_data

class ScannetDatasetTest():
    def __init__(self, root, num_classes=21, split='test', \
                 get_pixmeta=False, get_scene_point=False, \
                 frame_skip=1, num_proc=20):
        self.root = root
        self.split = split
        self.get_pixmeta = get_pixmeta
        self.point_from_depth = point_from_depth
        self.get_scene_point = get_scene_point
        self.num_classes = num_classes
        self.frame_skip = frame_skip
        self.pool = mp.Pool(num_proc)
        self.data_dir = os.path.join(self.root, '%s'%self.split)
        self.scene_list = glob.glob(os.path.join(self.data_dir, 'scene*'))
        print('[Dataset Test Info] Found ', len(self.scene_list), ' scenes under ', self.data_dir)

    def __len__(self):
        return len(self.scene_list)


    def __getitem__(self, index):
        start_time = time.time()
        scene_dir = self.scene_list[index]
        scene_name = os.path.basename(scene_dir)
        scene_point = None
        if self.get_scene_point is True:
            scene_point = np.load(os.path.join(scene_dir, scene_name+'.npy'))

        pixel_meta_list = list()
        color_img_list = list() 
        depth_img_list = list() 
        _, color_imgs, pixcoord_imgs, pixmeta_imgs = \
            dataset_utils.get_list(scene_dir=scene_dir, 
                     label_dir='pseudo_pose_label', color_dir='pseudo_pose_color',\
                     pixcoord_dir='pseudo_pose_pixel_coord', pixmeta_dir='pseudo_pose_pixel_meta',\
                     label_ext='png', color_ext='jpg', pixcoord_ext='npz', pixmeta_ext='npz')
        if len(color_imgs) > 0:
            data_dir = zip(color_imgs[::self.frame_skip], \
                           depth_imgs[::self.frame_skip], normal_imgs[::self.frame_skip], \
                           pixcoord_imgs[::self.frame_skip], pixmeta_imgs[::self.frame_skip], pose_txts[::self.frame_skip])
            get_scene_data = partial(dataset_utils.get_test_scene_data, 
                                     self.get_pixmeta)
            res = self.pool.map(get_scene_data, data_dir)
            print(scene_name) 
            color_img_list, pixel_meta_list = map(list, zip(*res))
        
        print('get {} time {}'.format(scene_name, time.time()-start_time))
        scene_data = {
            'scene_name': scene_name, 
            'scene_point': scene_point, 
            'num_view': len(color_img_list),
            'pixel_meta_list': pixel_meta_list,
            'color_img_list': color_img_list, 
        }    
        return scene_data


if __name__ == "__main__":
    from utils import vis_utils
    d = ScannetDatasetTest(root = '/tmp3/hychiang/scannetv2_data', num_classes=21, \
                           get_pixmeta=True, get_scene_point=True, frame_skip=1)
    start_time = time.time()
    for i in range(5):
        scene_data = d[i]
        print(scene_data['scene_name'], len(scene_data['pixel_meta_list']), \
              len(scene_data['color_img_list']), len(scene_data['depth_img_list']), \
              len(scene_data['coord_img_list']), len(scene_data['valid_coord_idx_list']))
    print('Load all valid scenes with pixel meta, step=5: ', time.time()-start_time)
