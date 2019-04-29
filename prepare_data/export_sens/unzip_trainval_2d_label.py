import argparse
import os, sys
import shutil
import zipfile
import multiprocessing as mp

# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--label-filt', dest='unzip_label_filt', action='store_true')
parser.add_argument('--label', dest='unzip_label', action='store_true')
parser.add_argument('--instance-filt', dest='unzip_instance_filt', action='store_true')
parser.add_argument('--instance', dest='unzip_instance', action='store_true')
parser.add_argument('--num_proc', type=int, default=5, help='number of process for exporting sens in parallel')

parser.set_defaults(unzip_label=False, unzip_label_filt=False, unzip_instance=False, unzip_instance_filt=False)
opt = parser.parse_args()
print(opt)



def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

def unzip_file(zip_file, output_dir):
    if(not os.path.isfile(zip_file)):
        print_error(zip_file+' not exist') 
#    print('unzip '+ zip_file)
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(output_dir)
    zip_ref.close()

def unzip_2d_label(scene):
    print(scene)
    output_dir = os.path.join(opt.scannet_path, scene)
    if(opt.unzip_label):
        label_zip_file = os.path.join(opt.scannet_path, scene, scene+'_2d-label.zip')
        unzip_file(label_zip_file, output_dir)
    if(opt.unzip_label_filt):
        label_filt_zip_file = os.path.join(opt.scannet_path, scene, scene+'_2d-label-filt.zip')
        unzip_file(label_filt_zip_file, output_dir)
    if(opt.unzip_instance):
        instance_zip_file = os.path.join(opt.scannet_path, scene, scene+'_2d-instance.zip')
        unzip_file(instance_zip_file, output_dir)
    if(opt.unzip_instance_filt):
        instance_filt_zip_file = os.path.join(opt.scannet_path, scene, scene+'_2d-instance-filt.zip')
        unzip_file(instance_filt_zip_file, output_dir)

def main():

    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    scenes.sort()
    print('Found %d scenes' % len(scenes))

    pool = mp.Pool(opt.num_proc)
    pool.map(unzip_2d_label, scenes)
    #for s in scenes:
    #    unzip_2d_label(s)


if __name__ == '__main__':
    main()

