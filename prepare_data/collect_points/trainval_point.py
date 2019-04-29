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
g_label_names = scannet_utils.g_label_names
g_label_ids = scannet_utils.g_label_ids

SCANNET_DIR = None     # '/tmp3/hychiang/ScanNet.v2/ScanNet/scans/'  
SCENE_NAMES = None     # [line.rstrip() for line in open('./Benchmark/scannetv2_train.txt')]
LABEL_MAP_FILE = None  # './scannetv2-labels.combined.tsv' 
OUTPUT_FOLDER = None   # 'scans_train'
LOG_FILE = 'log.txt'
LOG_FOUT = None

def collect_one_scene_data_label(scene_name, out_filename):
    # read label mapping file
    label_map = scannet_utils.read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to='nyu40id')

    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(SCANNET_DIR, scene_name)
    # Read segmentation label 
    seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
    seg_to_verts, num_verts = scannet_utils.read_segmentation(seg_filename)
    # Read Instances segmentation label
    agg_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name))
    object_id_to_segs, label_to_segs = scannet_utils.read_aggregation(agg_filename)
    # Raw points in XYZRGBA
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    #points = pc_util.read_ply_rgba(ply_filename)
    points = pc_utils.read_ply_rgba_normal(ply_filename)

    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated 
    for label, segs in label_to_segs.items():
        # convert scannet raw label to nyu40 label (1~40), 0 for unannotated, 41 for unknown
        label_id = label_map[label]

        # only evaluate 20 class in nyu40 label
        # map nyu40 to 1~21, 0 for unannotated, unknown and not evalutated
        if label_id in g_label_ids: # IDS for 20 classes in nyu40 for evaluation (1~21)
            eval_label_id = g_label_ids.index(label_id)
        else: # IDS unannotated, unknow or not for evaluation go to unannotate label (0)
            eval_label_id = g_label_names.index('unannotate')
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = eval_label_id
    #for i in range(20):
    #    print(label_ids[i])

    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id

    points = np.delete(points, 6, 1) #  only RGB, ignoring A
    label_ids = np.expand_dims(label_ids, 1)
    instance_ids = np.expand_dims(instance_ids, 1)
    #print(points.shape, label_ids.shape, instance_ids.shape)
     # order is critical, do not change the order
    data = np.concatenate((points, instance_ids, label_ids), 1)
    #print(data.shape)
    #for i in range(20):
    #    print(data[i, 10])
    np.save(out_filename, data)
    log_string(scene_name+' save to '+out_filename+', with data: point:'+str(points.shape)+', label:'+str(label_ids.shape)+', instance label:'+str(instance_ids.shape)+', data:'+str(data.shape))

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
        collect_one_scene_data_label(scene_name, out_file)
    except Exception as e:
        log_string(scene_name+'ERROR!!')
        log_string(str(e))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_list", required=True, help="scannet split scene list, e.g. ./Benchmark/scannetv2_train.txt")
    parser.add_argument("--label_map_file", required=True, help="scannet label mapping file , e.g. ./scannetv2-labels.combined.tsv")
    parser.add_argument("--scannet_dir",required=True,  help="scannet data dir, e.g. {path/to/scannet/data/dir}/scans or {path/to/scannet/data/dir}/scans_test")
    parser.add_argument("--output_dir", required=True, help="output dir (folder), e.g. ./scans_train")
    parser.add_argument("--num_proc", required=False, type=int, default=30, help="number of parallel process, default is 30")
    args = parser.parse_args()
    LABEL_MAP_FILE = args.label_map_file
    SCENE_NAMES = [line.rstrip() for line in open(args.scene_list)]
    SCANNET_DIR = args.scannet_dir 
    OUTPUT_FOLDER = args.output_dir
    
    print('***  Total Scene in list: ', len(SCENE_NAMES))
    print('***  Read Label Mapping File from: ', LABEL_MAP_FILE)
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
