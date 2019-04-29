import os
import sys
import time
import math
import glob
import argparse
import imageio
import numpy as np
import  multiprocessing as mp
from PIL import Image

import scannet_util
from human_sorting import human_sort
g_label_ids = scannet_util.g_label_ids

DATA_DIR = None
LOG_FILE = 'render_label_log.txt'

def barycentric_weight(pixcoord, pixmeta):
    # pixel coord: x, y, z
    # pixel meta: v1, v2, v3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z
    pixcoord = pixcoord.reshape(-1, 3)
    pixmeta = pixmeta.reshape(-1, 12)
    mesh_vertex = pixmeta[:,0:3].astype(np.int32)
    # barycentry interpolation
    area = np.cross(pixmeta[:,6:9]- pixmeta[:,3:6], pixmeta[:,9:12]- pixmeta[:,3:6])  # cross(v2-v1, v3-v1)
    pa = np.cross(pixmeta[:,3:6]- pixcoord[:,0:3], pixmeta[:,6:9]- pixcoord[:,0:3])  # cross(v1-p, v2-p)
    pb = np.cross(pixmeta[:,6:9]- pixcoord[:,0:3], pixmeta[:,9:12]- pixcoord[:,0:3])  # cross(v2-p, v3-p)
    pc = np.cross(pixmeta[:,9:12]- pixcoord[:,0:3], pixmeta[:,3:6]- pixcoord[:,0:3])  # cross(v3-p, v1-p)
    #print(area[0], pa[0], pb[0], pc[0])
    area = np.linalg.norm(area, axis=1, keepdims=True) # area = length of outer product vector
    pa = np.linalg.norm(pa, axis=1, keepdims=True) # area = length of outer product vector
    pb = np.linalg.norm(pb, axis=1, keepdims=True) # area = length of outer product vector
    pc = np.linalg.norm(pc, axis=1, keepdims=True) # area = length of outer product vector
    
    v1w = pb / (area+1e-8)
    v2w = pc / (area+1e-8)
    v3w = pa / (area+1e-8)
    v1w[np.isnan(v1w)] = 0.
    v2w[np.isnan(v2w)] = 0.
    v3w[np.isnan(v3w)] = 0.

    return mesh_vertex, v1w, v2w, v3w


def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = scannet_util.create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    imageio.imwrite(filename, vis_image)


def render(scene_dir):
    # get pixel coordinates files 
    scene_pixel_coord = os.path.join(DATA_DIR, scene_dir, "pseudo_pose_pixel_coord")
    if not os.path.exists(scene_pixel_coord):
        print('[ERROR] Not found scene scene pixel coordinates direcotry {}'.format(scene_pixel_coord))
        raise
    else:
        pixcoord_npzs = human_sort(glob.glob(scene_pixel_coord+'/*.npz'))
    # get pixel meta files 
    scene_pixel_meta = os.path.join(DATA_DIR, scene_dir, "pseudo_pose_pixel_meta")
    if not os.path.exists(scene_pixel_meta):
        print('[ERROR] Not found scene scene pixel meta direcotry {}'.format(scene_pixel_meta))
        raise
    else:
        pixmeta_npzs = human_sort(glob.glob(scene_pixel_meta+'/*.npz'))
    assert len(pixcoord_npzs) == len(pixmeta_npzs), 'file number not equal:{}, {}'.format(len(pixcoord_npzs), len(pixmeta_npzs))

    # mkdir render_label directory 
    scene_render_label = os.path.join(DATA_DIR, scene_dir, "pseudo_pose_label")
    if not os.path.exists(scene_render_label):
        os.makedirs(scene_render_label) 
    # render label to render_label directory
    for i in range(len(pixcoord_npzs)):
        # load pixel coordinate and meta
        pixmeta_fname = os.path.splitext(os.path.basename(pixmeta_npzs[i]))[0]
        pixcoord_fname = os.path.splitext(os.path.basename(pixcoord_npzs[i]))[0]
        assert pixmeta_fname == pixcoord_fname, 'file name not equal:{}, {}'.format(pixmeta_fname, pixcoord_fname)
        pixmeta = np.load(pixmeta_npzs[i])['meta']
        pixcoord = np.load(pixcoord_npzs[i])['pixcoord']
        assert pixmeta.shape[0:2] == pixcoord.shape[0:2]
        # load scene points label
        scene_pt_label = np.load(os.path.join(DATA_DIR, scene_dir, scene_dir+'.npy'))[:, 10]

        # get closest vertex of each pixel
        mesh_vertex, v1w, v2w, v3w = barycentric_weight(pixcoord, pixmeta)
        max_weight_idx = np.argmax(np.concatenate([v1w, v2w, v3w], axis=-1), axis=-1)
        max_weight_vertex = mesh_vertex[np.arange(len(max_weight_idx)), max_weight_idx]
        # get vertex label
        image_label = scene_pt_label[max_weight_vertex].astype(np.int32)
        nodepth_idx = np.where(pixcoord.reshape(-1, 3) == np.array([0, 0, 0]))[0]
        image_label[nodepth_idx] = 0
        # visualize render label
        image_label_vis = np.zeros(image_label.shape, dtype=np.int32)
        for i in range(len(image_label)):
            image_label_vis[i] = g_label_ids[image_label[i]]
        # save label image for training
        image_label = image_label.reshape(pixcoord.shape[0:2])
        label_img = Image.fromarray(image_label.astype(np.uint8))
        label_png = os.path.join(scene_render_label, pixmeta_fname+'.png')
        label_img.save(label_png)
        # save label image for visualization
        image_label_vis = image_label_vis.reshape(pixcoord.shape[0:2])
        image_label_vis_png = os.path.join(scene_render_label, pixmeta_fname+'.vis.png')
        visualize_label_image(image_label_vis_png, image_label_vis)
    print('[INFO] Finish {}'.format(scene_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_list", required=True, help="scannet split scene list, e.g. ./Benchmark/scannetv2_train.txt")
    parser.add_argument("--data_dir",required=True,  help="data dir, e.g. ../data/scannet_frames_train")
    parser.add_argument("--num_proc", required=False, type=int, default=30, help="number of parallel process, default is 30")
    args = parser.parse_args()
    DATA_DIR = args.data_dir 
    SCENE_NAMES = [line.rstrip() for line in open(args.scene_list)]
    

    print('***  Data Directory: ', DATA_DIR)
    print('***  Find ', len(SCENE_NAMES),' scenes in scene txt')
    print('***  Rendering label from pixel meta. Start in 5 seconds ***')

    LOG_FOUT = open(os.path.join(DATA_DIR, LOG_FILE),'w')
    #time.sleep(5)

    print('*** GO ***')
    pool = mp.Pool(args.num_proc)
    pool.map(render, SCENE_NAMES)
    LOG_FOUT.close()

