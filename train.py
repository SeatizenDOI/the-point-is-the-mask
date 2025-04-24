
from argparse import Namespace, ArgumentParser

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.ASVManager import ASVManager
from src.UAVManager import UAVManager
from src.TileManager import TileManager
from src.utils.lib_tools import print_header


from src.training.main import main_launch_training

from inference import parse_args as inference_parse_args
from inference import main as inference_main


def parse_args() -> Namespace:

    parser = ArgumentParser(prog="The point is the mask training", description="Workflow to WSSS training on UAV orthophoto using ASV prediction.")

    # Input.
    parser.add_argument("-cp", "--config_path", default="./config.json", help="Path to the config file.")
    parser.add_argument("-ep", "--env_path", default="./.env", help="Path to the env file.")

    return parser.parse_args()

def main(opt: Namespace) -> None:

    print_header()
    
    # Initialize parser.
    cp = ConfigParser(opt)

    # Initialize path manager.
    pm = PathManager(cp.output_path)
    pm.setup(cp)

    # Create coarse annotations.
    asv_manager = ASVManager(cp, pm)
    asv_manager.create_coarse_annotations()

    # Download uav orthophoto
    uav_manager = UAVManager(cp, pm)

    # Create tile and image and annotations
    tile_manager = TileManager(cp, pm)
    if tile_manager.create_tiles_and_annotations(uav_manager):
        test_images_list = tile_manager.convert_tiff_to_png(uav_manager.get_default_crs())
        tile_manager.convert_tiff_to_png_annotations(test_images_list)
        tile_manager.verify_if_annotation_tiles_contains_valid_values(asv_manager.get_classes_mapping())

    # First training.
    main_launch_training(cp, pm.train_folder, asv_manager.get_classes_mapping())

    # Plot result.

    # Inference with segrefiner.

    # Second training

    # Plot result.


    #

    inference_args = Namespace(
        enable_folder=False, enable_session=True, enable_csv=False, 
        path_folder='input', 
        path_session='input/20231202_REU-TROU-DEAU_UAV-01_01_ortho.tif', 
        path_csv_file=None, 
        path_segmentation_model='segmentation_model/quantile99-segmentation_model-ce0-dice1/checkpoint-5358', 
        path_geojson='config/emprise_lagoon.geojson', 
        horizontal_overlap=0.5, 
        vertical_overlap=0.5, 
        tile_size=512, 
        geojson_crs='EPSG:4326', 
        underwater_color_correction=False, 
        path_output='./output2', 
        index_start='0', 
        clean=True
    )

if __name__ == "__main__":
    opt = parse_args()
    main(opt)