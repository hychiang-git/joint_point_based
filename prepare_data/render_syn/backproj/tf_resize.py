import numpy as np
from PIL import Image
import tensorflow as tf

from utils import scannet_util
from utils import pc_util


coord_img = tf.placeholder(tf.float32, shape=(2, 480, 640, 3))
color_img = tf.placeholder(tf.float32, shape=(2, 480, 640, 3))
coord_img_resized = tf.image.resize_images(
                        coord_img,
                        (120, 160),
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        align_corners=True,
                        )
color_img_resized = tf.image.resize_images(
                        color_img,
                        (120, 160),
                        method=tf.image.ResizeMethod.BILINEAR,
                        align_corners=True,
                        )


coord_img_1 = np.expand_dims(np.load('/tmp3/hychiang/scannetv2_preprocess/train/scene0141_01/pixel_coord/000020.npz')['pixcoord'], axis=0)
coord_img_2 = np.expand_dims(np.load('/tmp3/hychiang/scannetv2_preprocess/train/scene0141_02/pixel_coord/000080.npz')['pixcoord'], axis=0)
color_img_1 = np.expand_dims(np.array(Image.open('/tmp3/hychiang/scannetv2_preprocess/train/scene0141_01/render_color/000020.jpg')), axis=0)
color_img_2 = np.expand_dims(np.array(Image.open('/tmp3/hychiang/scannetv2_preprocess/train/scene0141_02/render_color/000080.jpg')), axis=0)

print(coord_img_1)
print(coord_img_1.shape, coord_img_2.shape, color_img_1.shape, color_img_2.shape)
feed_dict = {coord_img:np.concatenate([coord_img_1, coord_img_2], axis=0), 
             color_img:np.concatenate([color_img_1, color_img_2], axis=0)}
sess = tf.Session()
coord_img_resized_val, color_img_resized_val = sess.run([coord_img_resized, color_img_resized], feed_dict=feed_dict)
coord_img_val = np.concatenate([coord_img_1, coord_img_2], axis=0) 
color_img_val = np.concatenate([color_img_1, color_img_2], axis=0)

pc_util.write_ply_rgb(coord_img_val[0].reshape(-1,3), color_img_val[0].reshape(-1, 3), 'dataset_vis/tf_reshape_pt_orisize_1.ply')
pc_util.write_ply_rgb(coord_img_val[1].reshape(-1,3), color_img_val[1].reshape(-1, 3), 'dataset_vis/tf_reshape_pt_orisize_2.ply')
pc_util.write_ply_rgb(coord_img_resized_val[0].reshape(-1,3), color_img_resized_val[0].reshape(-1, 3), 'dataset_vis/tf_reshape_pt_resize_1.ply')
pc_util.write_ply_rgb(coord_img_resized_val[1].reshape(-1,3), color_img_resized_val[1].reshape(-1, 3), 'dataset_vis/tf_reshape_pt_resize_2.ply')
