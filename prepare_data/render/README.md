# render images from ply files in ScanNet v2 dataset 

We use camera pose (extrinsic paramters) from SfM/Synthesis to render images and record all the pixel 3D coordinates.

For each camera pose, we record following information for each pixel:
1. color
2. depth
3. normal
4. 3D coordinate
5. triangle vertices and their 3D coordinate where color, depth and normal are rendered from.

Some codes are borrowed from [liu115/mesh2color_voxel](https://github.com/liu115/mesh2color_voxel)

## Built With

* [Xtensor](https://github.com/QuantStack/xtensor) - For dumping c++ array into npy, npz
* [Tinyply](https://github.com/ddiakopoulos/tinyply) - For reading ply files

## To Build
```
    $ cd render_scene
    // synthetic camera poses
    $ make syn_pose
    // Stuctural from Motion (SfM) camera poses
    $ make sfm_pose
```


## Example usage 

```
    // synthetic camera poses 
    $./syn_pose dataset_dir scene_list num_thread
        // dataset_dir, e.g. scannet_data/val/
        // scene_list, e.g. scannetv2_val.txt, (optional)
        // num_thread, e.g. 10,  (optional default=1)


    // Stuctural from Motion (SfM) camera poses
    $./sfm_pose dataset_dir scene_list num_thread
        // dataset_dir, e.g. scannet_data/val/
        // scene_list, e.g. scannetv2_val.txt, (optional)
        // num_thread, e.g. 10,  (optional default=1)
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

