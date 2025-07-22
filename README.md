<p align="center">
  <a href="https://doi.org/10.5281/zenodo.15455743"><img src="https://zenodo.org/badge/967978876.svg" alt="DOI"></a>
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

This repository is the official implementation of "The point is the mask: scaling coral reef segmentation with weak supervision" [ICCV-2025 Oral Presentation].

We present a weakly supervised framework that uses underwater imagery to train aerial segmentation models for scalable coral reef monitoring with minimal annotation effort.

## Abstract

Monitoring coral reefs at large spatial scales remains an open challenge, essential for assessing ecosystem health and informing conservation efforts. 
While drone-based aerial imagery offers broad spatial coverage, its limited resolution makes it difficult to reliably distinguish fine classes like coral morphotypes.
At the same time, obtaining pixel-level annotations over large spatial extents is costly and labor-intensive, limiting the scalability of existing deep learning segmentation methods to aerial data.
We present a multi-scale weakly supervised semantic segmentation framework that addresses this challenge by transferring fine-scale ecological information from underwater imagery to aerial data. 
Our method enables large-scale coral reef mapping from drone imagery with minimal manual annotation, combining classification-based supervision, spatial interpolation and self-distillation techniques. 
We demonstrate the efficacy of the approach, enabling large-area segmentation of coral morphotypes and illustrating its flexibility in integrating new classes.
This study presents a scalable, cost-effective methodology for high-resolution reef monitoring, combining low-cost data collection, weakly supervised deep learning and multi-scale remote sensing.

![LM_EM_alternate drawio]([https://github.com/ananthu-aniraj/masking_strategies_bias_removal/assets/50333505/462a5653-0e43-443f-836f-6fe6db09a723](https://github.com/SeatizenDOI/the-point-is-the-mask/blob/master/config/figure_schema.pdf))


## Installation

To ensure a consistent environment for all users, this project uses a Conda environment defined in a `requirements.yml` file. Follow these steps to set up your environment:

1. **Install Conda:** If you do not have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Create the Conda Environment:** Navigate to the root of the project directory and run the following command to create a new environment from the `requirements.yml` file:
   ```bash
   conda env create -f requirements.yml
   ```

3. **Activate the Environment:** Once the environment is created, activate it using:
   ```bash
   conda activate segment_env
   ```

4. **Download the need data:** First download a [model checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) to <path/to/checkpoint>. For example, download the default sam_vit_h:
    ```bash
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./models/sam_base_model/sam_vit_h_4b8939.pth
    ```

5. **Env file:** You need a `.env` file at root folder with inside:
    ```txt
    HUGGINGFACE_TOKEN=YOUR_TOKEN
    ```


5. **Troubleshooting:** 

-- Fastgeodis setup. 

if 
```bash
Traceback (most recent call last):
  File "/home/bioeos/Documents/project_hub/the-point-is-the-mask/train.py", line 14, in <module>
    from inference import main as inference_main
  File "/home/bioeos/Documents/project_hub/the-point-is-the-mask/inference.py", line 8, in <module>
    from src.inference.ModelManager import ModelManager
  File "/home/bioeos/Documents/project_hub/the-point-is-the-mask/src/inference/ModelManager.py", line 14, in <module>
    from ..utils.sam_refiner import sam_refiner
  File "/home/bioeos/Documents/project_hub/the-point-is-the-mask/src/utils/sam_refiner.py", line 5, in <module>
    from .sam_utils import prepare_image, extract_bboxes_expand, extract_points, extract_mask
  File "/home/bioeos/Documents/project_hub/the-point-is-the-mask/src/utils/sam_utils.py", line 6, in <module>
    import FastGeodis
  File "/home/bioeos/miniconda3/envs/segment_env/lib/python3.12/site-packages/FastGeodis/__init__.py", line 34, in <module>
    import FastGeodisCpp
ImportError: /home/bioeos/miniconda3/envs/segment_env/lib/python3.12/site-packages/FastGeodisCpp.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE

```

Use `cp dependencies/FastGeodisCpp.cpython-312-x86_64-linux-gnu.so /home/bioeos/miniconda3/envs/segment_env/lib/python3.12/site-packages/` 

Replace `/home/bioeos/` by your path.

-- ⚠️ **Not use anymore but keep track for next dev.**

You need to install `sudo apt-get install libgdal-dev`. You need to match the libgdal version with gdal pip package version.

if
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

## How to use it.

To reproduce **the point is the mask**, you just need to launch the training part. All the configuration are already setup.


### Training.

You can update the configuration with the `config.json` file.

To launch the script, `python train.py`



### Inference.

Launch `python inference.py -h`

Launch `python inference.py -efol -c -is_seatizen_session`




# Optional information - use to generate tiles and serve with tms-server.


## Create your segmentation dataset.

Launch gdal with : `docker run --rm -it --user 1000:1000 -v .:/app ghcr.io/osgeo/gdal:latest`

and then go into /app and run `./apply_gdal.sh`

Use this file to gather your tiles : https://github.com/SeatizenDOI/tms-server/blob/master/utils/merge_tiles.py

Run `python merge_tiles.py -p /home/bioeos/Documents/project_hub/tms-server/bathy/tiles/`

After you get your global_tile folder you can follow the [tms-server README](https://github.com/SeatizenDOI/tms-server/blob/master/README.md)


## Runner

python main.py -efol -c && docker run --rm -it --user 1000:1000 -v .:/app ghcr.io/osgeo/gdal:latest sh -c "cd /app && ./apply_gdal.sh"
