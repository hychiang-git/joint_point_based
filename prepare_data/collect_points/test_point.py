import os
import sys
import time
import argparse
import json
import numpy as np
import  multiprocessing as mp
from functools import partial
from utils import pc_util
from utils import scannet_util
g_label_names = scannet_util.g_label_names
g_label_ids = scannet_util.g_label_ids

SCANNET_DIR = None     # '/tmp3/hychiang/ScanNet.v2/ScanNet/scans/'  
SCENE_NAMES = None     # [line.rstrip() for line in open('./Benchmark/scannetv2_train.txt')]
OUTPUT_FOLDER = None   # 'scans_train'
LOG_FILE = 'log.txt'
LOG_FOUT = None

def collect_one_scene_data(scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(SCANNET_DIR, scene_name)
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = pc_util.read_ply_rgba_normal(ply_filename)

    points = np.delete(points, 6, 1) #  only RGB, ignoring A
    data = points
    np.save(out_filename, data)
    log_string(scene_name+' save to '+out_filename+', point:'+str(points.shape)+', data:'+str(data.shape))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def preprocess_scenes(scene_name):
    log_string(scene_name)
    try:
        out_dir = os.path.join(OUTPUT_FOLDER, scene_name) # scene0000_00/scene0000_00.npy
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_file = os.path.join(out_dir, scene_name+'.npy') # scene0000_00/scene0000_00.npy
        collect_one_scene_data(scene_name, out_file)
    except Exception as e:
        log_string(scene_name+'ERROR!!')
        log_string(str(e))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_list", required=True, help="scannet split scene list, e.g. ./Benchmark/scannetv2_train.txt")
    parser.add_argument("--scannet_dir",required=True,  help="scannet data dir, e.g. {path/to/scannet/data/dir}/scans or {path/to/scannet/data/dir}/scans_test")
    parser.add_argument("--output_dir", required=True, help="output dir (folder), e.g. ./scans_train")
    parser.add_argument("--num_proc", required=False, type=int, default=30, help="number of parallel process, default is 30")
    args = parser.parse_args()
    SCENE_NAMES = [line.rstrip() for line in open(args.scene_list)]
    SCANNET_DIR = args.scannet_dir 
    OUTPUT_FOLDER = args.output_dir
    
    print('***  Total Scene in list: ', len(SCENE_NAMES))
    print('***  ScanNet Data Directory: ', SCANNET_DIR)
    print('***  Output Directory: ', OUTPUT_FOLDER)
    print('***  NUM of Processes to parallel: ', args.num_proc)
    print('***  Extract points (Vertex XYZ, RGB, NxNyNx, Label, Instance-label) parallel in 5 Seconds***')

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    LOG_FOUT = open(os.path.join(OUTPUT_FOLDER, LOG_FILE),'w')
    time.sleep(5)

    print('*** GO ***')

    
    pool = mp.Pool(args.num_proc)
    pool.map(preprocess_scenes, SCENE_NAMES)

    LOG_FOUT.close()
