import argparse
import os, sys
import numpy as np
import skimage.transform as sktf
import imageio
import shutil
import multiprocessing as mp

''' 
    params 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='', help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=320, help='export image width')
parser.add_argument('--output_image_height', type=int, default=240, help='export image height')
parser.add_argument('--num_proc', type=int, default=5, help='number of process for exporting sens in parallel')

parser.set_defaults(export_depth_images=False, export_label_images=False)
opt = parser.parse_args()
if opt.export_label_images:
    assert opt.label_map_file != ''
print(opt)


''' 
    tools for export and conver label 
'''
label_map = None
g_label_names = None 
g_label_ids = None
if opt.export_label_images:
    try:
        sys.path.append('../utils')
        import scannet_utils
    except:
        print('Failed to import ScanNet code toolbox util')
        sys.exit(-1)
    label_map = scannet_utils.read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40id')
    g_label_names = scannet_utils.g_label_names
    g_label_ids = scannet_utils.g_label_ids

''' 
    tools for export .sens
'''
try:
    from SensorData import SensorData
except:
    print('Failed to import SensorData (from ScanNet code toolbox)')
    sys.exit(-1)




def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

# from https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/2d_helpers/convert_scannet_label_image.py
def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    # only evaluate 20 class in nyu40 label
    # map nyu40 to 1~21, 0 for unannotated, unknown and not evalutated
    for scannet_label, nyu40_label in label_mapping.items():
        if nyu40_label in g_label_ids: # IDS for 20 classes in nyu40 for evaluation (1~21)
            eval_label = g_label_ids.index(nyu40_label)
        else: # IDS unannotated, unknow or not for evaluation go to unannotate label (0)
            eval_label = g_label_names.index('unannotate')
        mapped[image==scannet_label] = eval_label
    return mapped.astype(np.uint8)


def export(scene):
    sens_file = os.path.join(opt.scannet_path, scene, scene + '.sens')
    if not os.path.isfile(sens_file):
        print_error('Error: sens path %s does not exist' % sens_file)
    
    output_scene_path = os.path.join(opt.scannet_path, scene)
    print(output_scene_path, sens_file)
    ''' pose and color '''
    output_pose_path = os.path.join(opt.scannet_path, scene, 'pose')
    if not os.path.isdir(output_pose_path):
        os.makedirs(output_pose_path)
    output_color_path = os.path.join(opt.scannet_path, scene, 'color')
    if not os.path.isdir(output_color_path):
        os.makedirs(output_color_path)

    ''' depth '''
    if opt.export_depth_images:
        output_depth_path = os.path.join(opt.scannet_path, scene, 'depth')
        if not os.path.isdir(output_depth_path):
            os.makedirs(output_depth_path)
    ''' label '''
    label_path = os.path.join(opt.scannet_path, scene, opt.label_type)
    if opt.export_label_images and not os.path.isdir(label_path):
        print_error('Error: using export_label_images option but label path %s does not exist' % label_path)
    if opt.export_label_images:
        output_label_path = os.path.join(opt.scannet_path, scene, 'mapped_'+opt.label_type)
        if opt.export_label_images and not os.path.isdir(output_label_path):
            os.makedirs(output_label_path)

    ## read and export
    #sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), len(scenes), scenes[i]))
    #sys.stdout.flush()
    sd = SensorData(sens_file)
    #sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
    #sys.stdout.flush()

    sd.export_intrinsics(output_scene_path)
    sd.export_poses(output_pose_path, frame_skip=opt.frame_skip)
    sd.export_color_images(output_color_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
    if opt.export_depth_images:
        sd.export_depth_images(output_depth_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)

    if opt.export_label_images:
        for f in range(0, len(sd.frames), opt.frame_skip):
            label_file = os.path.join(label_path, str(f) + '.png')
            image = np.array(imageio.imread(label_file))
            image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0, preserve_range=True, mode='constant')
            mapped_image = map_label_image(image, label_map)
            #print(np.min(mapped_image), np.max(mapped_image))
            imageio.imwrite(os.path.join(output_label_path, str(f) + '.png'), mapped_image)

def main():


    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    scenes.sort()
    print('Found %d scenes' % len(scenes))

    pool = mp.Pool(opt.num_proc)
    pool.map(export, scenes)
    #for s in scenes:
    #    export(s)


if __name__ == '__main__':
    main()

