## Joint Model

### To Train
```
usage: train.py [-h] [--save_freq SAVE_FREQ] [--max_epoch MAX_EPOCH]
                [--num_gpus NUM_GPUS] [--gpu_bsize GPU_BSIZE]
                [--slow_start_step SLOW_START_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --save_freq SAVE_FREQ
                        Save model frequency (every n epoches) [default: 5]
  --max_epoch MAX_EPOCH
                        Epoch to run [default: 501]
  --num_gpus NUM_GPUS   Number of GPU [default: 1]
  --gpu_bsize GPU_BSIZE
                        Batch size in a GPU [default: 4]
  --slow_start_step SLOW_START_STEP
                        Smaller learning rate for before slow_start_step
                        [default: 0]
```

### To Test

```
usage: test3d.py [-h] --restore_model RESTORE_MODEL --split SPLIT
                 [--save_feature] [--data_dir DATA_DIR]
                 [--use_image USE_IMAGE] [--batch_size BATCH_SIZE]
                 [--from_scene FROM_SCENE] [--to_scene TO_SCENE]

optional arguments:
  -h, --help            show this help message and exit
  --restore_model RESTORE_MODEL
                        path to testing model
  --split SPLIT         train val test split
  --save_feature        to save image feature
  --data_dir DATA_DIR   data directory
  --use_image USE_IMAGE
                        percentage of images to back project onto the 3D scene
  --batch_size BATCH_SIZE
                        batch_size
  --from_scene FROM_SCENE
                        the start index of all scenes
  --to_scene TO_SCENE   the end index of all scenes
```
