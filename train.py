from argparse import Namespace, ArgumentParser

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.ASVManager import ASVManager
from src.UAVManager import UAVManager
from src.TileManager import TileManager
from src.utils.lib_tools import print_header


from src.training.main import main_launch_training
from src.utils.training_step import TrainingStep

from inference import main_raster as inference_main

from src.evaluation.main_evaluation import perform_evalutation


def parse_args() -> Namespace:

    parser = ArgumentParser(prog="The point is the mask training", description="Workflow to WSSS training on UAV orthophoto using ASV prediction.")

    # Config.
    parser.add_argument("-cp", "--config_path", default="./config.json", help="Path to the config file.")
    parser.add_argument("-ep", "--env_path", default="./.env", help="Path to the env file.")
    parser.add_argument("-oe", "--only_evaluation", action="store_true", help="Only perform evalution. Skip all the rest.")


    return parser.parse_args()

def main(opt: Namespace) -> None:

    print_header()
    
    # Initialize parser.
    cp = ConfigParser(opt)

    # Initialize path manager.
    pm = PathManager(cp.output_path)
    pm.setup(cp)

    # Perform only evaluation.
    if opt.only_evaluation:
        perform_evalutation(pm, cp, cp.model_path_refine)
        return

    # Create coarse annotations.
    asv_manager = ASVManager(cp, pm)
    asv_manager.create_coarse_annotations()

    # Download uav orthophoto
    uav_manager = UAVManager(cp, pm)

    # Coarse - Create tile and image and annotations
    tile_manager = TileManager(cp, pm)
    if tile_manager.create_tiles_and_annotations_coarse(uav_manager):
        test_images_list = tile_manager.convert_tiff_to_png(uav_manager.get_default_crs(), TrainingStep.COARSE)
        tile_manager.convert_tiff_to_png_annotations(test_images_list, TrainingStep.COARSE)
        tile_manager.verify_if_annotation_tiles_contains_valid_values(asv_manager.get_classes_mapping(), TrainingStep.COARSE)

    uav_manager.generate_csv_uav_sessions_for_inference()

    # First training.
    if cp.model_path_coarse == None:
        first_model_path = main_launch_training(cp, pm.coarse_train_folder, asv_manager.get_classes_mapping(), TrainingStep.COARSE)
    else:
        first_model_path = cp.model_path_coarse

    # Inference with segrefiner.
    if not pm.uav_prediction_refine_raster_folder.exists() or len(list(pm.uav_prediction_refine_raster_folder.iterdir())) == 0:
        inference_args = Namespace(
            enable_folder=False, enable_session=False, enable_csv=True, 
            path_folder=None, path_session=None, path_csv_file=pm.uav_csv, 
            path_segmentation_model=first_model_path, 
            path_geojson=cp.list_geojson_to_keep_inference, 
            horizontal_overlap=0.5,   
            vertical_overlap=0.5, 
            tile_size=512, 
            underwater_color_correction=False, 
            path_output='./data', 
            index_start='0', 
            clean=True,
            use_sam_refiner=cp.with_sam_refiner, 
            path_sam_model=cp.path_sam_model,
            max_pixels_by_slice_of_rasters=800000000
        )
        inference_main(inference_args)
    else:
        print("\n\n------ [INFERENCE - Predictions rasters already exists] ------\n")

    # Refine - Create tile and image and annotations
    if tile_manager.create_tiles_and_annotations_refine(uav_manager):
        test_images_list = tile_manager.convert_tiff_to_png(uav_manager.get_default_crs(), TrainingStep.REFINE)
        tile_manager.convert_tiff_to_png_annotations(test_images_list, TrainingStep.REFINE)
        tile_manager.verify_if_annotation_tiles_contains_valid_values(asv_manager.get_classes_mapping(), TrainingStep.REFINE)

    # Second training.
    if cp.model_path_refine == None:
        second_model_path = main_launch_training(cp, pm.refine_train_folder, asv_manager.get_classes_mapping(), TrainingStep.REFINE)
    else:
        second_model_path = cp.model_path_refine

    # Plot result.
    perform_evalutation(pm, cp, second_model_path)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)