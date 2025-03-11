# Segmentation model

## Create your segmentatoin dataset.

Gather all your raster in one folder.

Launch gdal with : `docker run --rm -it --user 1000:1000 -v ./output:/app ghcr.io/osgeo/gdal:latest`

`gdaldem color-relief output/merged_prections/20231202_REU-TROU-DEAU_UAV-01_01_merged_predictions_color.tif color.txt output/merged_prections/20231202_REU-TROU-DEAU_UAV-01_01_merged_predictions_original_color.tif -alpha`
and then go into /app and run `./apply_gdal.sh`


Use this file to gather your tiles : https://github.com/SeatizenDOI/tms-server/blob/master/utils/merge_tiles.py
Go into utils/ and run `python merge_tiles.py -p /home/bioeos/Documents/project_hub/tms-server/bathy/tiles/`
