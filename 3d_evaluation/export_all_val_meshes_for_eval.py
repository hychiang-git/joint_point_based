# Example of the output format for evaluation for 3d semantic label and instance prediction.
# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh_for_evaluation.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

# python imports
import math
import glob
import time
import os, sys, argparse
import inspect
import json
import  multiprocessing as mp
from functools import partial

try:
    import numpy as np
except:
    print ("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import util
import util_3d

TASK_TYPES = {'label', 'instance'}

parser = argparse.ArgumentParser()
parser.add_argument('--scan_path', required=True, help='path to scannet for all scenes (e.g., data/ScanNet/v2/')
parser.add_argument('--scene_list', required=True, help='the tran/val/split txt file (e.g., scannetv2_val.txt')
parser.add_argument('--output_dir', required=True, help='the directory (folder) for output file')
parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
parser.add_argument('--type', required=True, help='task type [label or instance]')
opt = parser.parse_args()
assert opt.type in TASK_TYPES


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, label_map_file, type, output_file):
    label_map = util.read_label_mapping(opt.label_map_file, label_from='raw_category', label_to='nyu40id')
    mesh_vertices = util_3d.read_mesh_vertices(mesh_file)
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    if type == 'label':
        util_3d.export_ids(output_file, label_ids)
    elif type == 'instance':
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.iteritems():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
        util_3d.export_instance_ids_for_eval(output_file, label_ids, instance_ids)
    else:
        raise


def export_parallel(scene_name):
    mesh_file = os.path.join(opt.scan_path,scene_name, scene_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scene_name, scene_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scene_name, scene_name + '_vh_clean_2.0.010000.segs.json')
    output_file = os.path.join(opt.output_dir, scene_name+'.txt')
    assert os.path.isfile(mesh_file)==True, mesh_file
    assert os.path.isfile(agg_file)==True, agg_file
    assert os.path.isfile(seg_file)==True, seg_file
    export(mesh_file, agg_file, seg_file, opt.label_map_file, opt.type, output_file)
    print('\t evaluation file save to ', output_file)



def main():
    scenes_dir = os.listdir(opt.scan_path)
    print('*** Find ', len(scenes_dir), ' under ', opt.scan_path)
    scene_names = [line.rstrip() for line in open(opt.scene_list)]
    print('*** There are ', len(scene_names), ' in the scene list ', opt.scene_list)
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    print('*** Evaluation files will be exported parallel in 3 seconds')
    time.sleep(3)
    print('*** GO ***')

    pool = mp.Pool(25)
    pool.map(export_parallel, scene_names)

if __name__ == '__main__':
    main()
