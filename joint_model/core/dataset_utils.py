import glob
import os
import sys
import time
import imageio
import numpy as np
import multiprocessing as mp
from PIL import Image
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils import utils 
from utils import eulerangles as eua 

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

def get_list(scene_dir, label_dir, color_dir, depth_dir, normal_dir, pixcoord_dir, pixmeta_dir, pose_dir,\
             label_ext='png', color_ext='jpg', depth_ext='npz', normal_ext='npz', pixcoord_ext='npz', pixmeta_ext='npz', pose_ext='txt'):
    label_dir = os.path.join(scene_dir, label_dir)
    color_dir = os.path.join(scene_dir, color_dir)
    depth_dir = os.path.join(scene_dir, depth_dir)
    normal_dir = os.path.join(scene_dir, normal_dir)
    pose_dir = os.path.join(scene_dir, pose_dir)
    pixcoord_dir = os.path.join(scene_dir, pixcoord_dir)
    pixmeta_dir = os.path.join(scene_dir, pixmeta_dir)
    label_imgs = [l for l in glob.glob(os.path.join(label_dir, '*.'+label_ext)) if 'vis' not in l]
    color_imgs = glob.glob(os.path.join(color_dir, '*.'+color_ext))
    depth_imgs = glob.glob(os.path.join(depth_dir, '*.'+depth_ext))
    normal_imgs = glob.glob(os.path.join(normal_dir, '*.'+normal_ext))
    pixcoord_imgs = glob.glob(os.path.join(pixcoord_dir, '*.'+pixcoord_ext))
    pixmeta_imgs = glob.glob(os.path.join(pixmeta_dir, '*.'+pixmeta_ext))
    pose_txts = glob.glob(os.path.join(pose_dir, '*.'+pose_ext))
    assert len(label_imgs) == len(color_imgs) == len(depth_imgs) == len(normal_imgs) \
           == len(pixcoord_imgs) == len(pixmeta_imgs) == len(pose_txts), \
           '[Error] {}, Image length not equal: label {}, color {}, depth {}, normal {}, pixel coordinates {}, pixel meta {}, pose {}' \
           .format(scene_dir, len(label_imgs), len(color_imgs), len(depth_imgs), len(normal_imgs), \
                   len(pixcoord_imgs), len(pixmeta_imgs), len(pose_txts))
    label_imgs = utils.human_sort(label_imgs)
    color_imgs = utils.human_sort(color_imgs)
    depth_imgs = utils.human_sort(depth_imgs)
    normal_imgs = utils.human_sort(normal_imgs)
    pixcoord_imgs = utils.human_sort(pixcoord_imgs)
    pixmeta_imgs = utils.human_sort(pixmeta_imgs)
    pose_txts = utils.human_sort(pose_txts)
    return label_imgs, color_imgs, depth_imgs, normal_imgs, pixcoord_imgs, pixmeta_imgs, pose_txts

def get_image_sample(labelweights, image_path, depth_path, label_path, get_depth=True):
    # Get a random image, label and camera pose from scene
    color_img = Image.open(image_path)  # load color
    color_img = np.array(color_img)
    label_img = Image.open(label_path)  # load label
    label_img = np.array(label_img)
    label_img = np.expand_dims(label_img, axis=-1)
    if get_depth:
        depth_img = np.load(depth_path)['depth']  # load depth
    else:
        depth_img = np.zeros(label_img.shape)
    assert color_img.shape[0:2] == label_img.shape[0:2] == depth_img.shape[0:2],\
           'height, width not equal corlor {}, label {}, depth {}'\
           .format(color_img.shape, label_img.shape, depth_img.shape)
    height = color_img.shape[0]
    width = color_img.shape[1]

    pixel_weight = labelweights[label_img.reshape(-1)]  # get pixel weight
    pixel_weight = pixel_weight.reshape(height, width, 1)
    occ_image_label_set = set(label_img.reshape(-1)) 
    occ_image_label_set = occ_image_label_set - set({0,1,2})
    return color_img, depth_img, label_img, pixel_weight, occ_image_label_set

def get_depth_coord_sample(labelweights, image_path, depth_path, label_path, int_param, rotation, translation):
    color_img = Image.open(image_path)  # load color
    color_img = np.array(color_img)
    label_img = Image.open(label_path)  # load label
    label_img = np.array(label_img)
    label_img = np.expand_dims(label_img, axis=-1)
    # detph to camera xyz 
    depth_img = np.load(depth_path)['depth']  # load depth
    assert color_img.shape[0:2] == label_img.shape[0:2] == depth_img.shape[0:2],\
           'height, width not equal corlor {}, label {}, depth {}'\
           .format(color_img.shape, label_img.shape, depth_img.shape)
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    #pixel_z = np.array(depth_img) / 1000.0
    pixel_z = np.squeeze(depth_img, axis=-1)
    valid_idx = (pixel_z!=0)
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, width, width, endpoint=False), \
                                   np.linspace(0, height, height, endpoint=False))
    cx = int_param[0,2] * (width / (int_param[0,2]*2))
    fx = int_param[0,0] * (width / (int_param[0,2]*2))
    cy = int_param[1,2] * (height / (int_param[1,2]*2))
    fy = int_param[1,1] * (height / (int_param[1,2]*2))
    pixel_x = pixel_z * (pixel_x-cx) / fx 
    pixel_y = pixel_z * (pixel_y-cy) / fy 
    pixel_x = np.expand_dims(pixel_x, axis=-1) 
    pixel_y = np.expand_dims(pixel_y, axis=-1) 
    pixel_z = np.expand_dims(pixel_z, axis=-1) 
    pixel_xyz = np.concatenate((pixel_x, pixel_y, pixel_z), axis=-1)
    # depth world xyz without translation
    coord_img = pixel_xyz.reshape(-1, 3)
    valid_pts = np.where(coord_img[:, 2] != 0)[0]
    #print(valid_pts)
    coord_img = np.matmul(rotation, coord_img.T).T
    coord_xy_center = np.mean(coord_img[valid_pts, 0:2], axis=0)
    coord_img[valid_pts, 0:2] = coord_img[valid_pts, 0:2] - coord_xy_center
    # align 
    x_vec = np.array([[1.], [0.], [0.]]) 
    x_rotated = np.matmul(rotation, x_vec)
    x_rotated[2, 0] = 0.
    horizontal_shift = np.arccos(np.matmul(x_rotated.T, x_vec))
    if x_rotated[1,0] > 0:
        align_z = eua.euler2mat(z=-horizontal_shift, y=0, x=0)
    else:
        align_z = eua.euler2mat(z=horizontal_shift, y=0, x=0)
    coord_img = np.matmul(align_z, coord_img.T).T
    coord_img[valid_pts, 2] = coord_img[valid_pts, 2] + translation[2]
    # reshape to image shape
    coord_img = coord_img.reshape(height, width, 3)
    coord_img = np.concatenate([coord_img, color_img], axis=-1)
    # coord weight
    pixel_weight = labelweights[label_img.reshape(-1)]  # get pixel weight
    coord_weight = pixel_weight.reshape((height, width, 1)) * \
                   np.expand_dims(valid_idx.astype(np.int32), axis=-1)
    #print('from depth', coord_img.shape, label_img.shape, coord_weight.shape, valid_idx.shape)
    return coord_img, label_img, coord_weight, valid_idx 

def get_coord_sample(labelweights, image_path, pixcoord_path, label_path, int_param, rotation, translation):
    color_img = Image.open(image_path)  # load color
    color_img = np.array(color_img)
    label_img = Image.open(label_path)  # load label
    label_img = np.array(label_img)
    label_img = np.expand_dims(label_img, axis=-1)
    # detph to camera xyz 
    pixcoord_img = np.load(pixcoord_path)['pixcoord']  # load depth
    assert color_img.shape[0:2] == label_img.shape[0:2] == pixcoord_img.shape[0:2],\
           'height, width not equal corlor {}, label {}, pixel coord {}'\
           .format(color_img.shape, label_img.shape, pixcoord_img.shape)
    height = pixcoord_img.shape[0]
    width = pixcoord_img.shape[1]
    #pixel_z = np.array(depth_img) / 1000.0
    valid_idx = np.sum(pixcoord_img != np.array([0,0,0]), axis=-1) > 0
    coord_img = pixcoord_img.reshape(-1,3)
    valid_pts = np.where(coord_img!=np.array([0,0,0]))[0]
    #coord_img[valid_idx] = coord_img[valid_idx] - translation
    #coord_img = np.matmul(np.linalg.inv(rotation),coord_img[valid_idx].T)
    # depth world xyz without translation
    coord_xy_center = np.mean(coord_img[valid_pts, 0:2], axis=0)
    coord_img[valid_pts, 0:2] = coord_img[valid_pts, 0: 2] - coord_xy_center
    # align 
    x_vec = np.array([[1.], [0.], [0.]]) 
    x_rotated = np.matmul(rotation, x_vec)
    x_rotated[2, 0] = 0.
    horizontal_shift = np.arccos(np.matmul(x_rotated.T, x_vec))
    if x_rotated[1,0] > 0:
        align_z = eua.euler2mat(z=-horizontal_shift, y=0, x=0)
    else:
        align_z = eua.euler2mat(z=horizontal_shift, y=0, x=0)
    coord_img = np.matmul(align_z, coord_img.T).T
    # reshape to image shape
    coord_img = coord_img.reshape(height, width, 3)
    coord_img = np.concatenate([coord_img, color_img], axis=-1)
    # coord weight
    pixel_weight = labelweights[label_img.reshape(-1)]  # get pixel weight
    coord_weight = pixel_weight.reshape((height, width, 1)) * \
                   np.expand_dims(valid_idx.astype(np.int32), axis=-1)
    #print('from coord', coord_img.shape, label_img.shape, coord_weight.shape, valid_idx.shape)
    return coord_img, label_img, coord_weight, valid_idx 

def get_scene_data(labelweights, cam_int_param, get_depth, get_coord, get_pixmeta, point_from_depth, data_dir):
    label_dir, color_dir, depth_dir, normal_dir, pixcoord_dir, pixmeta_dir, pose_dir = data_dir
    ext_param = np.loadtxt(pose_dir)
    if np.any(np.isnan(ext_param)) or np.any(np.isinf(ext_param)):
        print(ext_param)
        print(pose_dir)
        raise 
    color_img, depth_img, pixel_label, pixel_weight, label_set = \
        get_image_sample(labelweights, color_dir, depth_dir, label_dir, get_depth)
    #print(color_img.shape, depth_img.shape, pixel_label.shape, pixel_weight.shape)
    if get_coord is True:
        translation = ext_param[0:3, 3] # [0,3], [1,3], [2,3]
        rotation = ext_param[0:3, 0:3]
        if point_from_depth is True:
            coord_img, coord_label, coord_weight, valid_coord_idx = \
                get_depth_coord_sample(self.labelweights, \
                                        color_dir, depth_dir, label_dir, \
                                        cam_int_param, rotation, translation)
        else: # load from pixcoord
            coord_img, coord_label, coord_weight, valid_coord_idx = \
                get_coord_sample(labelweights, \
                                  color_dir, pixcoord_dir, label_dir, \
                                  cam_int_param, rotation, translation)
        if get_pixmeta is True:
            pixel_world_coord = np.load(pixcoord_dir)['pixcoord']  # load depth
            pixel_meta = np.load(pixmeta_dir)['meta']
            pixel_meta = np.concatenate([pixel_world_coord, pixel_meta], axis=-1)
        else:
            pixel_meta = np.zeros((color_img.shape[0], color_img.shape[1], 15))
    else:
        coord_img = np.zeros((color_img.shape[0], color_img.shape[1], 6))
        coord_label = np.zeros(pixel_label.shape)
        coord_weight = np.zeros(pixel_weight.shape)
        valid_coord_idx = np.zeros((pixel_weight.shape[0], pixel_weight.shape[1]))
        pixel_meta = np.zeros((color_img.shape[0], color_img.shape[1], 15))

    return color_img, depth_img, pixel_label, pixel_weight, \
           coord_img, coord_label, coord_weight, valid_coord_idx, \
           pixel_meta

