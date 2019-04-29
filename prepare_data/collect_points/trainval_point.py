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

''' 
    params 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--label_map_file', default='', help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument("--num_proc", required=False, type=int, default=30, help="number of parallel process, default is 30")
opt = parser.parse_args()


def collect_point_data(scene_name):
    # read label mapping file
    label_map = scannet_utils.read_label_mapping(opt.label_map_file, label_from='raw_category', label_to='nyu40id')

    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(opt.scannet_path, scene_name)
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
    print('Extract points (Vertex XYZ, RGB, NxNyNx, Label, Instance-label)')
    
    pool = mp.Pool(opt.num_proc)
    pool.map(preprocess_scenes, scenes)

if __name__=='__main__':
    main()
