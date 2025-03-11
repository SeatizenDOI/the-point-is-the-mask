#!/bin/bash

COLOR_FOLDER=./rasters_color
rm -rf $COLOR_FOLDER
mkdir -p $COLOR_FOLDER


TILES_FOLDER=./tiles
rm -rf $TILES_FOLDER
mkdir -p $TILES_FOLDER

STOP=$(ls merged_rasters | wc -l)
for ((i=0; i<$STOP; i++))
do

    FILE="merged_rasters/group_${i}_cog.tif"
    if [ ! -f $FILE ]; then
        echo "File not found: ${FILE}"
        continue
    fi
    COLOR_OUTPUT_FILE="${COLOR_FOLDER}/group_${i}_color.tif"

    gdaldem color-relief $FILE color.txt $COLOR_OUTPUT_FILE -alpha

    TILE_OUTPUT_FOLDER="${TILES_FOLDER}/tiles_group_${i}_color"

    gdal2tiles --processes=12 -x -n -z 14-22 $COLOR_OUTPUT_FILE $TILE_OUTPUT_FOLDER

done