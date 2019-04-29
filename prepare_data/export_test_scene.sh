rm log/export_test.log

export () {
    python3 ./export_sens/prepare_test_scene.py \
        --scannet_path /tmp3/hychiang/ScanNet.v2/ScanNet/scans_test/ \
        --output_path /tmp3/hychiang/ScanNet.v2/ScanNet/scans_test_extracted/ \
        --output_image_width 640 \
        --output_image_height 480 \
        --export_depth_images \
        --frame_skip 10 \
        --from_scene $1 \
        --to_scene $2 \
        2>&1 | tee -a log/export_test.log
}

for i in {0..3}
do
    a=$(($i*25))
    b=$(($i*25+25))
    echo $a
    echo $b
    export $a $b & 
done
