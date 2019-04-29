import os
import sys
import csv
import time
import math
import glob
import struct
import shutil
import argparse
import numpy as np
#from PIL import Image
import imageio
import  multiprocessing as mp

DEPTH_SHIFT = 1000
DATA_DIR = None
RENDER_BINARY = None
SCENE_NAMES = None
LOG_FILE = 'render_log.txt'


def decode_bin(binfile, width, height, color_dir, depth_dir, normal_dir, pixcoord_dir, pixmeta_dir, vis_depth=True, vis_normal=True):
    fname = os.path.splitext(os.path.basename(binfile))[0]
    fin = open(binfile, 'rb')
    height = struct.unpack('I', fin.read(4))[0]
    width = struct.unpack('I', fin.read(4))[0]

    # decode color
    color_fname = os.path.join(color_dir, fname+".jpg")
    color_byte_size = struct.unpack('Q', fin.read(8))[0]
    color = b''.join(struct.unpack('c'*color_byte_size, fin.read(color_byte_size)))
    color = np.frombuffer(color, dtype=np.int32).reshape((height, width, 3))
    color = color.astype(np.uint8)
    imageio.imwrite(color_fname, color)
   
    # decode depth
    depth_fname_npz = os.path.join(depth_dir, fname+".npz")
    depth_byte_size = struct.unpack('Q', fin.read(8))[0]
    depth = b''.join(struct.unpack('c'*depth_byte_size, fin.read(depth_byte_size)))
    depth = np.frombuffer(depth, dtype=np.float64).reshape((height, width, 1))
    np.savez_compressed(depth_fname_npz, depth=depth)
    if vis_depth is True:
        depth_fname_png = os.path.join(depth_dir, fname+".png")
        depth = np.array(depth).reshape(height, width)
        depth = 255 * (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
        depth = depth.astype(np.uint8)
        imageio.imwrite(depth_fname_png, depth)

    # decode normal 
    normal_fname_npz = os.path.join(normal_dir, fname+".npz")
    normal_byte_size = struct.unpack('Q', fin.read(8))[0]
    normal = b''.join(struct.unpack('c'*normal_byte_size, fin.read(normal_byte_size)))
    normal = np.frombuffer(normal, dtype=np.float64).reshape((height, width, 3))
    np.savez_compressed(normal_fname_npz, normal=normal)
    if vis_normal is True:
        normal_fname_png = os.path.join(normal_dir, fname+".png")
        normal = 255 * ( normal + 1. ) / 2
        normal = normal.astype(np.uint8)
        imageio.imwrite(normal_fname_png, normal)

    # decode pixel coordinate 
    pixcoord_fname_npz = os.path.join(pixcoord_dir, fname+".npz")
    pixcoord_byte_size = struct.unpack('Q', fin.read(8))[0]
    pixcoord = b''.join(struct.unpack('c'*pixcoord_byte_size, fin.read(pixcoord_byte_size)))
    pixcoord = np.frombuffer(pixcoord, dtype=np.float64).reshape((height, width, 3))
    np.savez_compressed(pixcoord_fname_npz, pixcoord=pixcoord)

    # decode pixel meta data 
    pixmeta_fname_npz = os.path.join(pixmeta_dir, fname+".npz")
    mv_byte_size = struct.unpack('Q', fin.read(8))[0] # get mesh vertiex index
    mesh_vertex = b''.join(struct.unpack('c'*mv_byte_size, fin.read(mv_byte_size)))
    mesh_vertex = np.frombuffer(mesh_vertex, dtype=np.int32).reshape((height, width, 3))
    mvcoord_byte_size = struct.unpack('Q', fin.read(8))[0] # get mesh vertiex index
    mvcoord = b''.join(struct.unpack('c'*mvcoord_byte_size, fin.read(mvcoord_byte_size)))
    mvcoord = np.frombuffer(mvcoord, dtype=np.float64).reshape((height, width, 9))
    pixmeta = np.concatenate([mesh_vertex, mvcoord], axis=-1)
    np.savez_compressed(pixmeta_fname_npz, meta=pixmeta)
 
def read_camera_parameter(camera_parameter_txt):
    with open(camera_parameter_txt, 'r') as f:
        line_list = list()
        for line in f:
            line_arr = np.array(line.strip().split(' ')).astype(np.float32)
            line_list.append(line_arr)
        matrix = np.array(line_list)
    return matrix

def render(scene):
    start_time = time.time()
    scene_dir = os.path.join(DATA_DIR, scene)
    print('[INFO] Rendering ', scene_dir)
    scene_name = os.path.basename(scene_dir)
    scene_tmp_dir = os.path.join(scene_dir, "tmp")
    if not os.path.exists(scene_tmp_dir):
        os.makedirs(scene_tmp_dir) 
    scene_ply = os.path.join(scene_dir, scene_name+'_vh_clean_2.ply')
    if not os.path.exists(scene_ply):
        print('[ERROR] Not found scene ply file {}'.format(scene_ply))
        sys.exit()
    scene_pose_dir = os.path.join(scene_dir, "pose")
    if not os.path.exists(scene_pose_dir):
        print('[ERROR] Not found scene pose directory {}'.format(scene_pose_dir))
        sys.exit()
    scene_depth_intrinsic = os.path.join(scene_dir, "intrinsic_depth.txt")
    if not os.path.exists(scene_depth_intrinsic):
        print('[ERROR] Not found scene depth intrinsic {}'.format(scene_depth_intrinsic))
        sys.exit()
    scene_depth_extrinsic = os.path.join(scene_dir, "extrinsic_depth.txt")
    if not os.path.exists(scene_depth_extrinsic):
        print('[ERROR] Not found scene depth extrinsic {}'.format(scene_depth_extrinsic))
        sys.exit()
    intrinsic = read_camera_parameter(scene_depth_intrinsic)
    width = math.ceil(intrinsic[0,2]*2); 
    height = math.ceil(intrinsic[1,2]*2);
    #print("[INFO] Image width:", width, ", Image height:", height, " from intrinsic.")
    if width > 620 and width < 660:
        width = 640
    if height > 460 and height < 500:
        height = 480
    if width != 640 or height != 480:
        print('[ERROR] Camera intrinsic not recognized height {}, width {}'.format(height, width))
        sys.exit()
    #print("[INFO] Normalized Image width:", width, ", Image height:", height)

    # call c++ rendering program
    start_render_time = time.time()
    cmd = RENDER_BINARY+' '+scene_ply+' '+ scene_depth_intrinsic+' '+ scene_depth_extrinsic+' '+ scene_pose_dir+' '+scene_tmp_dir
    print("[INFO] Execute rendering program for scene {}".format(scene_dir))
    os.system(cmd)
    print("[INFO] Takes ", time.time()-start_render_time, " to render ", scene_dir)

    # start decoding bin file
    start_decode_time = time.time()
    scene_render_depth = os.path.join(scene_dir, "render_depth")
    if not os.path.exists(scene_render_depth):
        os.makedirs(scene_render_depth) 
    scene_render_color = os.path.join(scene_dir, "render_color")
    if not os.path.exists(scene_render_color):
        os.makedirs(scene_render_color) 
    scene_render_normal = os.path.join(scene_dir, "render_normal")
    if not os.path.exists(scene_render_normal):
        os.makedirs(scene_render_normal) 
    scene_pixel_coord = os.path.join(scene_dir, "pixel_coord")
    if not os.path.exists(scene_pixel_coord):
        os.makedirs(scene_pixel_coord)
    scene_pixel_meta = os.path.join(scene_dir, "pixel_meta")
    if not os.path.exists(scene_pixel_meta):
        os.makedirs(scene_pixel_meta) 
    binfiles = glob.glob(scene_tmp_dir+'/*.bin')
    print("[INFO] decoding {}".format(scene_dir) )
    for binfile in binfiles:
        decode_bin(binfile, width, height, scene_render_color, scene_render_depth, scene_render_normal, scene_pixel_coord, scene_pixel_meta)
    print("[INFO] Takes ", time.time()-start_decode_time, " to decode ", scene_dir)

    print("[INFO] remove tmpe file: ", scene_tmp_dir)
    shutil.rmtree(scene_tmp_dir, ignore_errors=True)
    print("[INFO] Finish ", scene_dir, ", ", time.time()-start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_list", required=True, help="scannet split scene list, e.g. ./Benchmark/scannetv2_train.txt")
    parser.add_argument("--data_dir",required=True,  help="data dir, e.g. ../data/scannet_frames_train")
    parser.add_argument("--render_excutable",required=True,  help="the compiled rendering executable, e.g. {path/to}/render")
    parser.add_argument("--num_proc", required=False, type=int, default=30, help="number of parallel process, default is 30")
    args = parser.parse_args()
    DATA_DIR = args.data_dir 
    RENDER_BINARY = args.render_excutable
    if not os.path.exists(RENDER_BINARY):
        print('[ERROR] Not found rendering binary file (c++ executable) ', RENDER_BINARY)
    SCENE_NAMES = [line.rstrip() for line in open(args.scene_list)]
    

    print('***  Data Directory: ', DATA_DIR)
    print('***  Render ', len(SCENE_NAMES), ' scenes')
    print('***  Rendering depth, color, normal, pixel_meta from ply. Start in 5 seconds ***')

    LOG_FOUT = open(os.path.join(DATA_DIR, LOG_FILE),'w')
    time.sleep(5)

    print('*** GO ***')
    pool = mp.Pool(args.num_proc)
    pool.map(render, SCENE_NAMES)
    LOG_FOUT.close()

