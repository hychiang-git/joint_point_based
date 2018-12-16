import os
import imageio
import numpy as np
from PIL import Image
from collections import Counter

from utils import scannet_util
from utils import pc_util
g_label_ids = scannet_util.g_label_ids

def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = scannet_util.create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    imageio.imwrite(filename, vis_image)

def dump_point_cloud(scene_name, output_dir, coord_img, valid_coord_idx, coord_label=None, coord_weight=None, vis_weight_factor=40):
    ## dump color point clouds
    vpt_coord = coord_img[valid_coord_idx, 0:3]
    vpt_color = coord_img[valid_coord_idx, 3:6]
    pc_util.write_ply_rgb(vpt_coord, vpt_color, os.path.join(output_dir, scene_name+'_vpt.ply'))
    if coord_label is not None:
        # dump point clouds label
        color_map = scannet_util.create_color_palette()
        coord_label = np.squeeze(coord_label, axis=-1)
        coord_label_nyu = np.copy(coord_label)
        coord_label_cnt = Counter(coord_label.reshape(-1))
        for val_id, cnt in coord_label_cnt.items():
           coord_label_nyu[np.where(coord_label==val_id)] = g_label_ids[val_id]
        vpt_label_nyu = coord_label_nyu[valid_coord_idx]
        pc_util.write_ply_color(vpt_coord, vpt_label_nyu, os.path.join(output_dir, scene_name+'_vpt_label.obj'), len(color_map), color_map)
    if coord_weight is not None:
        # dump weight 
        vpt_weight_color = np.squeeze(coord_weight, axis=-1)

        vpt_weight_color = vis_weight_factor * np.expand_dims(vpt_weight_color, axis=-1).astype(np.uint8)
        vpt_weight_color = np.concatenate((vpt_weight_color, vpt_weight_color, vpt_weight_color), axis=-1)
        vpt_weight_color = vpt_weight_color[valid_coord_idx]
        pc_util.write_ply_rgb(vpt_coord, vpt_weight_color, os.path.join(output_dir, scene_name+'_vpt_weight.ply'))

def dump_images(scene_name, output_dir, color_img, depth_img=None, pixel_label=None, pixel_weight=None, vis_weight_factor=40):
    # dump color
    color_img = Image.fromarray(color_img)
    color_img.save(os.path.join(output_dir, scene_name+'_color.jpg'))
    if depth_img is not None:
        # dump depth 
        depth_img = np.squeeze(depth_img, axis=-1)
        depth_img = 255*(depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))
        depth_img = Image.fromarray(depth_img.astype(np.uint8))
        depth_img.save(os.path.join(output_dir, scene_name+'_depth.jpg'))
    if pixel_label is not None:
        # dump label 
        pixel_label = np.squeeze(pixel_label, axis=-1)
        pixel_label_nyu = np.copy(pixel_label)
        pixel_label_cnt = Counter(pixel_label.reshape(-1))
        for val_id, cnt in pixel_label_cnt.items():
            pixel_label_nyu[np.where(pixel_label==val_id)] = g_label_ids[val_id]
        visualize_label_image(os.path.join(output_dir, scene_name+'_label.png'), pixel_label_nyu)
    if pixel_weight is not None:
        # dump weight 
        weight_img = np.squeeze(pixel_weight, axis=-1)
        weight_img = weight_img * vis_weight_factor 
        weight_img = Image.fromarray(weight_img.astype(np.uint8))
        weight_img.save(os.path.join(output_dir, scene_name+'_weight.jpg'))
