<p align="center">
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/graphs/contributors"><img src="https://img.shields.io/github/contributors/SeatizenDOI/the-point-is-the-mask" alt="GitHub contributors"></a>
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/network/members"><img src="https://img.shields.io/github/forks/SeatizenDOI/the-point-is-the-mask" alt="GitHub forks"></a>
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/issues"><img src="https://img.shields.io/github/issues/SeatizenDOI/the-point-is-the-mask" alt="GitHub issues"></a>
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/blob/master/LICENSE"><img src="https://img.shields.io/github/license/SeatizenDOI/the-point-is-the-mask" alt="License"></a>
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/pulls"><img src="https://img.shields.io/github/issues-pr/SeatizenDOI/the-point-is-the-mask" alt="GitHub pull requests"></a>
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/stargazers"><img src="https://img.shields.io/github/stars/SeatizenDOI/the-point-is-the-mask" alt="GitHub stars"></a>
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/watchers"><img src="https://img.shields.io/github/watchers/SeatizenDOI/the-point-is-the-mask" alt="GitHub watchers"></a>
</p>
<div align="center">
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask">View framework</a>
  ·
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/issues">Report Bug</a>
  ·
  <a href="https://github.com/SeatizenDOI/the-point-is-the-mask/issues">Request Feature</a>
</div>

<div align="center">

# The Point Is The Mask

</div>

Weakly Supervised Semantic Segmentation using UAV and ASV on coral reef.


## Installation

To ensure a consistent environment for all users, this project uses a Conda environment defined in a `requirements.yml` file. Follow these steps to set up your environment:

1. **Install Conda:** If you do not have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Dependencies:** You need to install `sudo apt-get install libgdal-dev`. You need to match the libgdal version with gdal pip package version.

3. **Create the Conda Environment:** Navigate to the root of the project directory and run the following command to create a new environment from the `requirements.yml` file:
   ```bash
   conda env create -f requirements.yml
   ```

4. **Activate the Environment:** Once the environment is created, activate it using:
   ```bash
   conda activate segmentation_env
   ```
5. **Troubleshooting:** if
```bash
Exception: Python bindings of GDAL 3.10.2 require at least libgdal 3.10.2, but 3.8.4 was found
      [end of output] 
```

Use `sudo apt-get install libgdal-dev` to install libgdal.

Then `sudo apt-get info libgdal-dev` to get the version.

Mine is 3.8.4: 
```
Package: libgdal-dev
Version: 3.8.4+dfsg-3ubuntu3
Priority: extra
Section: universe/libdevel
Source: gdal
Origin: Ubuntu
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
```

Finally `pip install gdal==3.8.4`

If `ImportError: /home/bioeos/miniconda3/envs/segmentation_env/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libgdal.so.34)`

Use:

`sudo find / -name libstdc++.so.6` to find your local file.

`strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX` to check if the `version 'GLIBCXX_3.4.30'` is present.

Then:
```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/bioeos/miniconda3/envs/segmentation_env/lib/libstdc++.so
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/bioeos/miniconda3/envs/segmentation_env/lib/libstdc++.so.6
```


## Create your segmentation dataset.

Launch gdal with : `docker run --rm -it --user 1000:1000 -v .:/app ghcr.io/osgeo/gdal:latest`

and then go into /app and run `./apply_gdal.sh`

Use this file to gather your tiles : https://github.com/SeatizenDOI/tms-server/blob/master/utils/merge_tiles.py

Run `python merge_tiles.py -p /home/bioeos/Documents/project_hub/tms-server/bathy/tiles/`

After you get your global_tile folder you can follow the [tms-server README](https://github.com/SeatizenDOI/tms-server/blob/master/README.md)


## Runner

python main.py -efol -c && docker run --rm -it --user 1000:1000 -v .:/app ghcr.io/osgeo/gdal:latest sh -c "cd /app && ./apply_gdal.sh"



## Download the need data : 


First download a [model checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) to <path/to/checkpoint>. For example, download the default sam_vit_h:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O <path/to/checkpoint>
```

