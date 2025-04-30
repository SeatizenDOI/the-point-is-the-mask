
from argparse import Namespace, ArgumentParser

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.ASVManager import ASVManager
from src.UAVManager import UAVManager
from src.TileManager import TileManager
from src.utils.lib_tools import print_header


from src.training.main import main_launch_training
from src.training.training_step import TrainingStep

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

    uav_manager.generate_csv_uav_sessions_for_inference()

    # First training.
    first_model_path = main_launch_training(cp, pm.train_folder, asv_manager.get_classes_mapping(), TrainingStep.COARSE)

    # Inference with segrefiner.
    inference_args = Namespace(
        enable_folder=False, enable_session=False, enable_csv=True, 
        path_folder=None, path_session=None, path_csv_file=pm.uav_csv, 
        path_segmentation_model="./models/quantile99-segmentation_model-ce0-dice1/checkpoint-5358", 
        path_geojson="", 
        horizontal_overlap=0.5,   
        vertical_overlap=0.5, 
        tile_size=512, 
        geojson_crs='EPSG:4326', 
        underwater_color_correction=False, 
        path_output='./data', 
        index_start='0', 
        clean=True,
        use_sam_refiner=False, 
        path_sam_model='./models/sam_base_model/sam_vit_h_4b8939.pth'
    )
    inference_main(inference_args)


    # # Second training
    # main_launch_training(cp, pm.train_folder, asv_manager.get_classes_mapping())


    # Plot result.



if __name__ == "__main__":
    opt = parse_args()
    main(opt)