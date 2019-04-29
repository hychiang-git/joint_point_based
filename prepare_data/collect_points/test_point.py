import os
import sys
import time
import argparse
import json
import numpy as np
import  multiprocessing as mp
from functools import partial

sys.path.append('../utils')
import pc_utils
import scannet_utils

''' 
    params 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument("--num_proc", required=False, type=int, default=30, help="number of parallel process, default is 30")
opt = parser.parse_args()


def collect_point_data(scene_name):
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(opt.scannet_path, scene_name)
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = pc_utils.read_ply_rgba_normal(ply_filename)

    points = np.delete(points, 6, 1) #  only RGB, ignoring A
    data = points
    out_filename = os.path.join(data_folder, scene_name+'.npy') # scene0000_00/scene0000_00.npy
    np.save(out_filename, data)
    print(scene_name, ' points shape:', data.shape)


def preprocess_scenes(scene_name):
    try:
        collect_point_data(scene_name)
    except Exception as e:
        sys.stderr.write(scene_name+'ERROR!!')
        sys.stderr.write(str(e))
        sys.exit(-1)

def main():
    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    scenes.sort()
    print('Find %d scenes' % len(scenes))
    print('Extract points (Vertex XYZ, RGB, NxNyNx)')
    
    pool = mp.Pool(opt.num_proc)
    pool.map(preprocess_scenes, scenes)

if __name__=='__main__':
    main()
