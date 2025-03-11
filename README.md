# Segmentation model


6. **Troubleshooting:** if
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


## Create your segmentatoin dataset.

Gather all your raster in one folder.

Launch gdal with : `docker run --rm -it --user 1000:1000 -v ./output:/app ghcr.io/osgeo/gdal:latest`

`gdaldem color-relief output/merged_prections/20231202_REU-TROU-DEAU_UAV-01_01_merged_predictions_color.tif color.txt output/merged_prections/20231202_REU-TROU-DEAU_UAV-01_01_merged_predictions_original_color.tif -alpha`
and then go into /app and run `./apply_gdal.sh`


Use this file to gather your tiles : https://github.com/SeatizenDOI/tms-server/blob/master/utils/merge_tiles.py
Go into utils/ and run `python merge_tiles.py -p /home/bioeos/Documents/project_hub/tms-server/bathy/tiles/`
