# Export .sens files in ScanNet v2 dataset 

This directory shows exporting `.sens` files in ScanNet v2 dataset.
The codes are borrowed from [3DMV](https://github.com/angeladai/3DMV/blob/master/prepare_data/prepare_2d_data.py)


## Dependencies
* `numpy`, `imageio`, `scikit-image`, `opencv`
* depends on the [sens file reader from ScanNet](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py); should be placed in the same directory 
* if `export_label_images` flag is on:
	* depends on ScanNet util; should be placed in the same directory
	* assumes that label images are unzipped as scene `scene*/label*/*.png` 

## Example usage 

* To extract files from `.sens`. If you wish to export label for train/val set, please set the `label_map_file` flag and run `unzip_trainval_2d_label.py` first.
```
    python3 prepare_trainval_scene.py \
        --scannet_path /path/to/scannet/scans/ \
        --output_path /path/to/output/scans/ \
        --output_image_width 640 \
        --output_image_height 480 \
        // --export_depth_images \
        // --export_label_images \
        // --label_type label-filt \
        // --label_map_file /path/to/scannetv2-labels.combined.tsv \
        --frame_skip 20 \
        --num_proc 5 \
```

* To unzip 2d labels under `/scans/scene*/`
```
    python3 unzip_trainval_2d_label.py \
        --scannet_path /path/to/scannet/scans/ \
        --label-filt \
        //--label \
        //--instance-filt \
        //--instance \
```

<!--
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
--->

