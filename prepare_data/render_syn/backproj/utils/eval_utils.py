import numpy as np

def softmax(x, axis=-1):
    """Compute the softmax in a numerically stable way."""
    x = x - np.amax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return softmax_x

def barycentric_weight(pixmeta):
    # pixel meta: x, y, z, v1, v2, v3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z
    pixmeta = pixmeta.reshape(-1, 15)
    #print(pixmeta[0])
    mesh_vertex = pixmeta[:,3:6].astype(np.int32)
    # barycentry interpolation
    area = np.cross(pixmeta[:,9:12]- pixmeta[:,6:9], pixmeta[:,12:15]- pixmeta[:,6:9])  # cross(v2-v1, v3-v1)
    pa = np.cross(pixmeta[:,6:9]- pixmeta[:,0:3], pixmeta[:,9:12]- pixmeta[:,0:3])  # cross(v1-p, v2-p)
    pb = np.cross(pixmeta[:,9:12]- pixmeta[:,0:3], pixmeta[:,12:15]- pixmeta[:,0:3])  # cross(v2-p, v3-p)
    pc = np.cross(pixmeta[:,12:15]- pixmeta[:,0:3], pixmeta[:,6:9]- pixmeta[:,0:3])  # cross(v3-p, v1-p)
    #print(area[0], pa[0], pb[0], pc[0])
    area = np.linalg.norm(area, axis=1, keepdims=True) # area = length of outer product vector
    pa = np.linalg.norm(pa, axis=1, keepdims=True) # area = length of outer product vector
    pb = np.linalg.norm(pb, axis=1, keepdims=True) # area = length of outer product vector
    pc = np.linalg.norm(pc, axis=1, keepdims=True) # area = length of outer product vector
    #print(area[0], pa[0], pb[0], pc[0], pa[0]+pb[0]+pc[0])
    
    v1w = pb / (area+1e-8)
    v2w = pc / (area+1e-8)
    v3w = pa / (area+1e-8)
    v1w[np.isnan(v1w)] = 0.
    v2w[np.isnan(v2w)] = 0.
    v3w[np.isnan(v3w)] = 0.

    return mesh_vertex, v1w, v2w, v3w

def get_recorder(num_classes, *args):
    eval_list = ['seen_class', 'correct_class', 'fp_class', 'fn_class']
    recorder = dict()
    for method in args:
        recorder[method] = dict()
        for eval_item in eval_list:
            recorder[method][eval_item] = [0.0 for _ in range(num_classes)]
    return recorder

def record(num_classes, recorder, **kargs):
    eval_list = ['seen_class', 'correct_class', 'fp_class', 'fn_class']
    for method, record in recorder.items():
        if (method+'_pred' in kargs) and (method+'_label' in kargs):
            pred = kargs[method+'_pred']
            label = kargs[method+'_label']
            for l in range(1, num_classes):   # don't care class 0 (unannotated, unknown)
                recorder[method]['seen_class'][l] += np.sum(label==l)    # number of ground truth
                recorder[method]['correct_class'][l] += np.sum(          \
                                                            (pred==l)  & \
                                                            (label==l) & \
                                                            (pred>0)   & \
                                                            (label>0)    \
                                                        ) # True positive
                recorder[method]['fp_class'][l] += np.sum(          \
                                                       (pred==l)  & \
                                                       (label!=l) & \
                                                       (pred>0)   & \
                                                       (label>0)    \
                                                   )      # False positive
                recorder[method]['fn_class'][l] += np.sum(          \
                                                       (pred!=l)  & \
                                                       (label==l) & \
                                                       (label>0)    \
                                                   )      # False negative

def evaluate_score(recorder):
    for method, record in recorder.items():
        # total accuracy for valid label (class 1~20)
        acc = np.sum(np.array(record['correct_class'][1:], dtype=np.float32)) / \
                     np.sum(np.array(record['seen_class'][1:],dtype=np.float32))
        # class accuracy (class 0~20) 
        class_acc = np.array(record['correct_class'], dtype=np.float32) / \
                             (np.array(record['seen_class'], dtype=np.float)+1e-6)
        # class-wise average accuracy (class 1~20) 
        class_acc_val = np.array(class_acc[1:]) # ignore class 0
        seen_class_val = np.array(record['seen_class'][1:]) # ignore class 0
        avg_acc = np.mean(class_acc_val[np.where(seen_class_val>0)]) # only avgerage for those seen class
        # class IoU (class 0~20) 
        class_iou_denom = (np.array(record['correct_class'], dtype=np.float32) + \
                           np.array(record['fp_class'], dtype=np.float32)+ \
                           np.array(record['fn_class'], dtype=np.float32)+1e-6)
        class_iou = np.array(record['correct_class'], dtype=np.float32) / class_iou_denom 
        # class-wise Mean IoU (class 1~20) 
        class_iou_val = np.array(class_iou[1:]) # ignore class 0
        avg_iou = np.mean(class_iou_val[np.where(seen_class_val>0)]) # only avgerage for those seen class 
        record['acc'] = acc
        record['avg_acc'] = avg_acc
        record['class_acc'] = class_acc
        record['avg_iou'] = avg_iou
        record['class_iou'] = class_iou

def log_score(recorder, num_classes ,log_func, list_method=False):
    acc_string = '          acc: '
    avg_class_acc_string = 'avg class acc: '
    avg_class_iou_string = 'avg class iou: '
    for method, record in recorder.items():
        acc_string = acc_string + method + ' ' + '{:.5f}'.format(record['acc'])+ ', '
        avg_class_acc_string = avg_class_acc_string + method + ' ' + '{:.5f}'.format(record['avg_acc'])+ ', '
        avg_class_iou_string = avg_class_iou_string + method + ' ' + '{:.5f}'.format(record['avg_iou'])+ ', '
        if list_method:
            per_class_str =  '---- ' + method + ' ----\n'
            for l in range(1, num_classes):
                per_class_str += '\tclass %d, acc: %f; iou: %f, seen: %f, tp: %f, fp: %f, fn %f\n' % \
                  (l, record['class_acc'][l], record['class_iou'][l], record['seen_class'][l], \
                   record['correct_class'][l], record['fp_class'][l], record['fn_class'][l])
            log_func(per_class_str)
    log_func(acc_string)
    log_func(avg_class_acc_string)
    log_func(avg_class_iou_string)
