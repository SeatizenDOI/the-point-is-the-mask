
from argparse import Namespace, ArgumentParser
from pathlib import Path

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.ASVManager import ASVManager
from src.UAVManager import UAVManager
from src.TileManager import TileManager
from src.utils.lib_tools import print_header

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="The point is the mask training", description="Workflow to WSSS training on UAV orthophoto using ASV prediction.")

    # Input.
    parser.add_argument("-cp", "--config_path", default="./config.json", help="Path to the config file.")

    

    return parser.parse_args()

def main(opt: Namespace) -> None:

    print_header()
    
    # Initialize parser.
    cp = ConfigParser(opt)

    # Initialize path manager.
    pm = PathManager(cp.get_output_path())
    pm.setup(cp)

    # Create coarse annotations.
    asv_manager = ASVManager(cp, pm)
    asv_manager.create_coarse_annotations()

    uav_manager = UAVManager(cp, pm)

    tile_manager = TileManager(cp, pm)

    tile_manager.create_tiles_and_annotations(uav_manager)
    test_images_list = tile_manager.convert_tiff_to_png(uav_manager.get_default_crs())
    tile_manager.convert_tiff_to_png_annotations(test_images_list)
    tile_manager.verify_if_annotation_tiles_contains_valid_values(asv_manager.get_classes_mapping())


if __name__ == "__main__":
    opt = parse_args()
    main(opt)