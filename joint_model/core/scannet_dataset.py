import pickle
import os
import sys
import glob
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
import dataset_utils

def rotate_point_cloud_z(xyz, normal):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    # move to center
    coordmax = np.max(xyz, axis=0)
    coordmin = np.min(xyz, axis=0)
    center = (coordmax + coordmin)/2
    xyz[:, 0:2] = xyz[:, 0:2] - center[0:2]
    # rotate
    xyz = np.dot(xyz, rotation_matrix)
    # shift to +x +y axis
    new_coordmin = np.min(xyz, axis=0)
    xyz[:, 0:2] = xyz[:, 0:2] + new_coordmin[0:2]

    # rotate normal
    normal = np.dot(normal, rotation_matrix)
    return xyz, normal

class ScannetDatasetTrain():
    def __init__(self, root, num_classes, vpoints=8192, spoints=16384, split='train', use_feature='feature', aug_z=True, dropout=True):
        self.vpoints = vpoints
        self.spoints = spoints
        self.root = root
        self.split = split
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.aug_z = aug_z
        self.dropout = dropout
        self.data_dir = os.path.join(self.root, '%s'%self.split)
        self.scene_list = glob.glob(os.path.join(self.data_dir, 'scene*'))
        print('[Dataset Train Info] Found ', len(self.scene_list), ' scenes under ', self.data_dir)
        self.labelweights = np.ones(self.num_classes)
        weight_file = os.path.join(self.root, 'class_weights.txt')
        if os.path.exists(weight_file):
            self.labelweights = dataset_utils.readClassesWeights(weight_file, self.num_classes)
        assert len(self.labelweights) == num_classes
        print('[Dataset Train Info] Training labelweights:\n', self.labelweights)
    def __len__(self):
        return len(self.scene_list)
    def __getitem__(self, index):
        scene_dir = self.scene_list[index]
        scene_name = os.path.basename(scene_dir)

        load_time = time.time()
        scene_data = np.load(os.path.join(scene_dir, scene_name+'.npy'))
        #print(os.path.join(scene_dir, scene_name+'.{}.npy'.format(self.use_feature)))
        scene_feature = np.load(os.path.join(scene_dir, scene_name+'.{}.npy'.format(self.use_feature)))
        load_time = time.time() - load_time

        scene_point = scene_data[:, 0:3]
        scene_color = scene_data[:, 3:6]
        scene_normal = scene_data[:, 6:9]
        scene_label = scene_data[:, 10].astype(np.int32) 
        if self.aug_z is True:
            scene_point, scene_normal = rotate_point_cloud_z(scene_point, scene_normal)
        coordmax = np.max(scene_point,axis=0)
        coordmin = np.min(scene_point,axis=0)

        batch_time = time.time()
        isvalid = False
        for i in range(10):
            curcenter = scene_point[np.random.choice(len(scene_label),1)[0],:]
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((scene_point>=(curmin-0.2))*(scene_point<=(curmax+0.2)),axis=1)==3
            cur_point = scene_point[curchoice,:]
            cur_color = scene_color[curchoice,:]
            cur_normal = scene_normal[curchoice,:]
            cur_feature = scene_feature[curchoice,:]
            cur_label = scene_label[curchoice]
            if len(cur_label)==0:
                continue
            mask = np.sum((cur_point>=curmin)*(cur_point<=curmax),axis=1)==3
            isvalid = (np.sum(cur_label*mask>0)/len(cur_label))>=0.7
            if isvalid:
                break
        scene_choice = np.random.choice(len(scene_label), self.spoints, replace=True)
        scene_smp_point = scene_point[scene_choice,:]
        scene_smp_color = scene_color[scene_choice,:]
        scene_smp_normal = scene_normal[scene_choice,:]
        scene_smp_feature = scene_feature[scene_choice,:]
        scene_smp_point = np.concatenate([scene_smp_point, scene_smp_normal, scene_smp_color, scene_smp_feature], axis=-1)
        choice = np.random.choice(len(cur_label), self.vpoints, replace=True)
        feature = cur_feature[choice,:]
        color = cur_color[choice,:]
        normal = cur_normal[choice,:]
        point = cur_point[choice,:]
        point = np.concatenate([point, normal, color, feature], axis=-1)

        label = cur_label[choice]
        mask = mask[choice]
        weight = self.labelweights[label]
        weight *= mask

        if self.dropout is True:
            dropout_ratio = np.random.random()*0.875 # 0-0.875
            drop_idx = np.where(np.random.random((point.shape[0]))<=dropout_ratio)[0]
            point[drop_idx,:] = point[0,:]
            label[drop_idx] = label[0]
            weight[drop_idx] *= 0

        return scene_name, scene_point, point, label, weight, scene_smp_point


class ScannetDatasetVal():
    def __init__(self, root, num_classes, vpoints=8192, spoints=16384, split='val', use_feature='feature', scene_padding=0, scene_stride=2):
        self.vpoints = vpoints
        self.spoints = spoints
        self.root = root
        self.split = split
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.scene_padding = scene_padding
        self.scene_stride = scene_stride
        self.data_dir = os.path.join(self.root, '%s'%self.split)
        self.scene_list = glob.glob(os.path.join(self.data_dir, 'scene*'))
        print('[Dataset Val Info] Found ', len(self.scene_list), ' scenes under ', self.data_dir)
        self.labelweights = np.ones(self.num_classes)
        weight_file = os.path.join(self.root, 'class_weights.txt')
        if os.path.exists(weight_file):
            self.labelweights = dataset_utils.readClassesWeights(weight_file, self.num_classes)
        assert len(self.labelweights) == num_classes
        print('[Dataset Val Info] Training labelweights:\n', self.labelweights)
    def __len__(self):
        return len(self.scene_list)
    def __getitem__(self, index):
        get_time = time.time()
        scene_dir = self.scene_list[index]
        scene_name = os.path.basename(scene_dir)

        scene_data = np.load(os.path.join(scene_dir, scene_name+'.npy'))
        #print(os.path.join(scene_dir, scene_name+'.{}.npy'.format(self.use_feature)))
        scene_feature = np.load(os.path.join(scene_dir, scene_name+'.{}.npy'.format(self.use_feature)))

        scene_point = scene_data[:, 0:3]
        scene_color = scene_data[:, 3:6]
        scene_normal = scene_data[:, 6:9]
        scene_label = scene_data[:, 10].astype(np.int32) 
        coordmax = np.max(scene_point,axis=0)
        coordmin = np.min(scene_point,axis=0)


        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        scene_list = list()
        point_list = list()
        label_list = list()
        weight_list = list()
        mask_list = list()
        for i in np.arange(-self.scene_padding, nsubvolume_x+self.scene_padding, self.scene_stride):
            for j in np.arange(-self.scene_padding, nsubvolume_y+self.scene_padding, self.scene_stride):
                curmin = coordmin+[i*1.5, j*1.5, 0]
                curmax = coordmin+[(i+1)*1.5, (j+1)*1.5, coordmax[2]-coordmin[2]]
                curchoice = np.sum((scene_point>=(curmin-0.2))*(scene_point<=(curmax+0.2)),axis=1)==3
                cur_point_indices = np.where(curchoice>0)[0]
                cur_point = scene_point[curchoice,:]
                cur_normal= scene_normal[curchoice,:]
                cur_color = scene_color[curchoice,:]
                cur_feature = scene_feature[curchoice,:]
                cur_label = scene_label[curchoice]
                cur_mask = np.sum((cur_point>=curmin)*(cur_point<=curmax),axis=1)==3
                if sum(cur_mask) < self.vpoints/10:
                    continue
                scene_choice = np.random.choice(len(scene_label), self.spoints, replace=True)
                scene_smp_point = scene_point[scene_choice,:]
                scene_smp_color = scene_color[scene_choice,:]
                scene_smp_normal = scene_normal[scene_choice,:]
                scene_smp_feature = scene_feature[scene_choice,:]
                scene_smp_point = np.concatenate([scene_smp_point, scene_smp_normal, scene_smp_color, scene_smp_feature], axis=-1)
                choice = np.random.choice(len(cur_label), self.vpoints, replace=True)
                point = cur_point[choice,:]
                normal = cur_normal[choice,:]
                color = cur_color[choice,:]
                feature = cur_feature[choice,:]
                point = np.concatenate([point, normal, color, feature], axis=-1)

                mask = cur_mask[choice] 
                label = cur_label[choice]
                weight = self.labelweights[label] * mask
                scene_list.append(np.expand_dims(scene_smp_point,0)) # 1xNx265 (256+3+3+3)
                point_list.append(np.expand_dims(point,0)) # 1xNx265 (256+3+3+3)
                label_list.append(np.expand_dims(label,0)) # 1xN
                weight_list.append(np.expand_dims(weight,0)) # 1xN
                mask_list.append(np.expand_dims(mask,0))
        assert len(scene_list) == len(point_list) == len(label_list) == len(weight_list) == len(mask_list)
        num_volume = len(point_list)
        scene_array = np.concatenate(tuple(scene_list),axis=0)
        point_array = np.concatenate(tuple(point_list),axis=0)
        label_array = np.concatenate(tuple(label_list),axis=0)
        weight_array = np.concatenate(tuple(weight_list),axis=0)
        mask_array = np.concatenate(tuple(mask_list),axis=0)

        get_time = time.time() - get_time
        print('get time:', get_time)
       
        res = {
            'scene_name': scene_name,
            'scene_point': scene_point,
            'num_volume': num_volume,
            'points': point_array,
            'labels': label_array,
            'weights': weight_array, 
            'masks': mask_array,
            'scene_smpt': scene_array
        }
        return res 

class ScannetDatasetTest():
    def __init__(self, root, num_classes, split='test', use_feature='feature', spoints=16384, scene_padding=0.5, scene_stride=0.3):
        self.root = root
        self.spoints = spoints
        self.split = split
        self.use_feature = use_feature
        self.scene_padding = scene_padding
        self.scene_stride = scene_stride
        self.num_classes = num_classes
        self.data_dir = os.path.join(self.root, '%s'%self.split)
        self.scene_list = glob.glob(os.path.join(self.data_dir, 'scene*'))
        print('[Dataset Test Info] Found ', len(self.scene_list), ' scenes under ', self.data_dir)
        self.labelweights = np.ones(self.num_classes)
        weight_file = os.path.join(self.root, 'class_weights.txt')
        if os.path.exists(weight_file):
            self.labelweights = dataset_utils.readClassesWeights(weight_file, self.num_classes)
        print('[Dataset Test Info] Training labelweights:\n', self.labelweights)
    def __len__(self):
        return len(self.scene_list)
    def __getitem__(self, index):
        scene_dir = self.scene_list[index]
        scene_name = os.path.basename(scene_dir)
        scene_data = np.load(os.path.join(scene_dir, scene_name+'.npy'))
        print(os.path.join(scene_dir, scene_name+'.{}.npy'.format(self.use_feature)))
        scene_feature = np.load(os.path.join(scene_dir, scene_name+'.{}.npy'.format(self.use_feature)))
        scene_point = scene_data[:, 0:3]
        scene_color = scene_data[:, 3:6]
        scene_normal = scene_data[:, 6:9]
        coordmax = np.max(scene_point,axis=0)
        coordmin = np.min(scene_point,axis=0)

        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        point_list = list()
        index_list = list()
        mask_list = list()
        scene_list = list()
        #avg_num_points = 0
        for i in np.arange(-self.scene_padding, nsubvolume_x+self.scene_padding, self.scene_stride):
            for j in np.arange(-self.scene_padding, nsubvolume_y+self.scene_padding, self.scene_stride):
                curmin = coordmin+[i*1.5, j*1.5, 0]
                curmax = coordmin+[(i+1)*1.5, (j+1)*1.5, coordmax[2]-coordmin[2]]
                curchoice = np.sum((scene_point>=(curmin-0.2))*(scene_point<=(curmax+0.2)),axis=1)==3
                index = np.where(curchoice>0)[0]
                point = scene_point[curchoice,:]
                normal= scene_normal[curchoice,:]
                color = scene_color[curchoice,:]
                feature = scene_feature[curchoice,:]
                mask = np.sum((point>=curmin)*(point<=curmax),axis=1)==3
                if sum(mask)==0 or len(point)<1:
                    continue
                scene_choice = np.random.choice(len(scene_point), self.spoints, replace=True)
                scene_smp_point = scene_point[scene_choice,:]
                scene_smp_color = scene_color[scene_choice,:]
                scene_smp_normal = scene_normal[scene_choice,:]
                scene_smp_feature = scene_feature[scene_choice,:]
                scene_smp_point = np.concatenate([scene_smp_point, scene_smp_normal, scene_smp_color, scene_smp_feature], axis=-1)

                point = np.concatenate([point, normal, color, feature], axis=-1)
                scene_list.append(np.expand_dims(scene_smp_point,0)) # 1xNx265 (256+3+3+3)
                point_list.append(np.expand_dims(point, 0)) # 1xNx265 (256+3+3+3)
                index_list.append(np.expand_dims(index, 0)) # 1xN
                mask_list.append(np.expand_dims(mask, 0)) # 1xN
                #print(point.shape, index.shape)
                #avg_num_points += len(point)
        assert len(point_list)
        num_volume = len(point_list)
        #avg_num_points /= num_volume
        #print('AVG NUM POINTS: ', avg_num_points) 
       
        res = {
            'scene_name': scene_name,
            'scene_point': scene_point,
            'num_volume': num_volume,
            'point_list': point_list,
            'pidx_list': index_list,
            'mask_list': mask_list,
            'scene_list': scene_list
        }
        return res 


if __name__=='__main__':
    from utils import vis_utils
    #d = ScannetDatasetTrain(root = '/tmp3/hychiang/scannetv2_data', num_classes=21, split='train', vpoints=8192, spoints=16384, use_feature='syn_0.5')
    #for i in range(10):
    #    scene_name, scene_point, ps,seg,smpw, scene_smpt = d[i]
    #    print('{}/{}, {}, {}, {}, {}, {}, {}'.format(i, len(d), scene_name, scene_point.shape, ps.shape, seg.shape, smpw.shape,scene_smpt.shape))
    ##    vis_utils.dump_point_cloud(
    ##        scene_name=scene_name, 
    ##        output_dir='../dataset_vis', 
    ##        points=ps, 
    ##        label=seg, 
    ##        weight=smpw
    ##    ) 
    #d = ScannetDatasetVal(root = '/tmp3/hychiang/scannetv2_data', num_classes=21, split='val', vpoints=8192, spoints=16384, use_feature='syn_0.1')
    #for i in range(10):
    #    scene_data = d[i]
    #    print('{}/{}, {}, {}, {}, {}, {}, {}, {}, {}'\
    #    .format(i, len(d), scene_data['scene_name'], scene_data['scene_point'].shape, scene_data['num_volume'], \
    #            scene_data['points'].shape, scene_data['labels'].shape, scene_data['weights'].shape, scene_data['masks'].shape, scene_data['scene_smpt'].shape))

    d = ScannetDatasetTest(root = '/tmp3/hychiang/scannetv2_data', num_classes=21, split='test', use_feature='syn_1.0')
    for i in range(5):
        scene_data = d[i]
        print('{}/{}, {}, {}, {}, {}, {}, {}, {}'\
        .format(i, len(d), scene_data['scene_name'], scene_data['scene_point'].shape, scene_data['num_volume'], \
                scene_data['point_list'][0].shape, scene_data['pidx_list'][0].shape,scene_data['mask_list'][0].shape, scene_data['scene_list'][0].shape))
    #exit()
