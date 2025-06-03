#!/bin/bash

COLOR_FOLDER=./output/rasters_color
rm -rf $COLOR_FOLDER
mkdir -p $COLOR_FOLDER


TILES_FOLDER=./output/tiles
rm -rf $TILES_FOLDER
mkdir -p $TILES_FOLDER

RASTER_FOLDER=./final_predictions_raster

for filename in $RASTER_FOLDER/*.tif;
do
    echo $filename;
    BASENAME=$(basename "$filename" .tif)

    if [ ! -f $filename ]; then
        echo "File not found: ${filename}"
        continue
    fi
    COLOR_OUTPUT_FILE="${COLOR_FOLDER}/${BASENAME}_color.tif"

    gdaldem color-relief $filename color.txt $COLOR_OUTPUT_FILE -alpha -co COMPRESS=LZW

    TILE_OUTPUT_FOLDER="${TILES_FOLDER}/tiles_${BASENAME}_color"

    gdal2tiles --processes=12 -x -n -z 10-22 $COLOR_OUTPUT_FILE $TILE_OUTPUT_FOLDER

done