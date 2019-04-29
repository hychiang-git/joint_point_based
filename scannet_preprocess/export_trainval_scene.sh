rm log/export_trainval.log

export () {
    python3 ./export_sens/prepare_trainval_scene.py \
        --scannet_path /tmp3/hychiang/ScanNet.v2/ScanNet/scans/ \
        --output_path /tmp3/hychiang/ScanNet.v2/ScanNet/scans_extracted/ \
        --output_image_width 640 \
        --output_image_height 480 \
        --export_depth_images \
        --frame_skip 20 \
        --from_scene $1 \
        --to_scene $2
        2>&1 | tee -a log/export_trainval.log
}

for i in {0..5}
do
    a=$(($i*300))
    b=$(($i*300+300))
    echo $a
    echo $b
    export $a $b & 
done
