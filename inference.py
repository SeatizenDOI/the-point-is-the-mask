import traceback
from pathlib import Path
from datetime import datetime
from argparse import Namespace, ArgumentParser

from src.inference.TileManager import TileManager
from src.inference.ModelManager import ModelManager
from src.inference.MosaicManager import MosaicManager
from src.inference.PathRasterManager import PathRasterManager
from src.inference.SeatizenSessionManager import SeatizenSessionManager
from src.utils.lib_tools import get_list_rasters, get_list_sessions

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="Segmentation Inference", description="Workflow to perform segmentation inference on UAV orthophoto.")

    # Input.
    arg_input = parser.add_mutually_exclusive_group(required=True)
    arg_input.add_argument("-efol", "--enable_folder", action="store_true", help="Work from a folder of orthophoto")
    arg_input.add_argument("-eses", "--enable_session", action="store_true", help="Work with one orthophoto")
    arg_input.add_argument("-ecsv", "--enable_csv", action="store_true", help="Work from csv with root_folder,ortho_name and if it is seatizen session, root_folder,session_name")

    # Path of input.
    parser.add_argument("-pfol", "--path_folder", default="/media/bioeos/E1/drone/serge_temp", help="Path to folder of session")
    parser.add_argument("-pses", "--path_session", default="/media/bioeos/E/drone/serge_drone_session/20231202_REU-TROU-DEAU_UAV-01_01/PROCESSED_DATA/PHOTOGRAMMETRY/odm_orthophoto/odm_orthophoto.tif", help="Path to the session")
    parser.add_argument("-pcsv", "--path_csv_file", default=None, help="Path to the csv file")

    # Model arguments.
    parser.add_argument("-psm", "--path_segmentation_model", default="models/SegForCoralBig-2025_05_20_27648-bs16_refine_b2", help="Path to semgentation model, currently only in local.")
    parser.add_argument("-pgeo", "--path_geojson", type=list, default=["./config/emprise_lagoon.geojson"], help="Path to geojson to crop ortho inside area. We can use multiple geojson")
    
    parser.add_argument("-ho", "--horizontal_overlap", type=float, default=0.5, help="Horizontal overlap between tiles.")
    parser.add_argument("-vo", "--vertical_overlap", type=float, default=0.5, help="Vertical overlap between tiles.")
    parser.add_argument("-ts", "--tile_size", type=int, default=512, help="Split Orthophoto into tiles.")
    parser.add_argument("-ucc", "--underwater_color_correction", action="store_true", help="Perform color corection on water.")

    # SAM arguments.
    parser.add_argument("-sam", "--use_sam_refiner", action="store_true", help="Refine prediction with SAMRefiner.")
    parser.add_argument("-psam", "--path_sam_model", default="./models/sam_base_model/sam_vit_h_4b8939.pth", help="Path to SAM refiner model.")

    # Output.
    parser.add_argument("-po" , "--path_output", default="./output2", help="Path of output")

    # Optional arguments.
    parser.add_argument("--is_seatizen_session", action="store_true", help="Instead of working with raster, we work with seatizen session.")
    parser.add_argument("-is", "--index_start", default="0", help="Choose from which index to start")
    parser.add_argument("-c", "--clean", action="store_true", help="Delete all previous file")
    parser.add_argument("--max_pixels_by_slice_of_rasters", type=int, default=800000000, help="Max pixels number into intermediate rasters to avoid RAM overload.")

    return parser.parse_args()

def main_raster(opt: Namespace) -> None:

    print("\n\n------ [INFERENCE - Working with rasters.] ------\n")
    print("\n\n------ [INFERENCE - Initialize managers.] ------\n")

    tile_manager = TileManager(opt)
    model_manager = ModelManager(opt)

    list_rasters = get_list_rasters(opt)
    index_start = int(opt.index_start) if opt.index_start.isnumeric() and int(opt.index_start) < len(list_rasters) else 0
    list_rasters = list_rasters[index_start:]
    rasters_fail = []
    for i, raster_path in enumerate(list_rasters):
        # Filter all files who are not rasters files
        if not raster_path.is_file() or raster_path.suffix != ".tif": continue

        print(f"\n\n--- {i+1}/{len(list_rasters)} - Working with {raster_path.stem}")        
        t_start = datetime.now()
        path_manager = PathRasterManager(opt.path_output, raster_path)

        # Clean if needed
        path_manager.clean() if opt.clean else path_manager.create_path()

        try:
            if opt.clean or path_manager.is_empty_cropped_folder():
                tile_manager.split_ortho_into_tiles(path_manager)

            if opt.clean or path_manager.is_empty_cropped_img_folder():
                tile_manager.convert_tiff_tiles_into_png(path_manager)

            if opt.clean or path_manager.is_empty_predictions_tiff_folder():
                model_manager.inference(path_manager)
            
            mosaic_manager = MosaicManager(path_manager, model_manager.get_id2label(), opt.max_pixels_by_slice_of_rasters)
            mosaic_manager.build_raster()
            
        except Exception as e:
            print(traceback.format_exc(), end="\n\n")
            rasters_fail.append(raster_path.name)
        finally:
            if opt.clean:
                path_manager.disk_optimize() 

        print(f"\n*\t Running time {datetime.now() - t_start}")

    # Stat
    print(f"\nEnd of process. On {len(list_rasters)} rasters, {len(rasters_fail)} fails.")
    if (len(rasters_fail)):
        [print("\t* " + session_name) for session_name in rasters_fail]  



def main_seatizen(opt: Namespace) -> None:

    print("\n\n------ [INFERENCE - Working with seatizen session.] ------\n")
    print("\n\n------ [INFERENCE - Initialize managers.] ------\n")

    tile_manager = TileManager(opt)
    model_manager = ModelManager(opt)

    list_sessions = get_list_sessions(opt)
    index_start = int(opt.index_start) if opt.index_start.isnumeric() and int(opt.index_start) < len(list_sessions) else 0
    list_sessions = list_sessions[index_start:]
    sessions_fail = []
    for i, session_path in enumerate(list_sessions):

        if not session_path.exists() or not session_path.is_dir(): 
            print(f"{session_path} not found, we continue.")
            continue
        
        session_manager = SeatizenSessionManager(session_path)
        raster_path = session_manager.orthophoto_path

        print(f"\n\n--- {i+1}/{len(list_sessions)} - Working with {raster_path.stem}")        
        t_start = datetime.now()
        path_manager = PathRasterManager(opt.path_output, raster_path)

        # Clean if needed
        path_manager.clean() if opt.clean else path_manager.create_path()

        try:
        #     if opt.clean or path_manager.is_empty_cropped_folder():
        #         tile_manager.split_ortho_into_tiles(path_manager)

        #     if opt.clean or path_manager.is_empty_cropped_img_folder():
        #         tile_manager.convert_tiff_tiles_into_png(path_manager)

        #     if opt.clean or path_manager.is_empty_predictions_tiff_folder():
        #         model_manager.inference(path_manager)
            
        #     mosaic_manager = MosaicManager(path_manager, model_manager.get_id2label(), opt.max_pixels_by_slice_of_rasters)
        #     mosaic_manager.build_raster()

        #     session_manager.move_prediction_raster(path_manager.final_merged_tiff_file, opt.path_segmentation_model.split("/")[-1])

            # TODO add metadata about raster prediction
            session_manager.create_resume_pdf(Path(opt.path_segmentation_model).name)
            
        except Exception as e:
            print(traceback.format_exc(), end="\n\n")
            sessions_fail.append(raster_path.name)
        finally:
            if opt.clean:
                path_manager.disk_optimize() 

        print(f"\n*\t Running time {datetime.now() - t_start}")

    # Stat
    print(f"\nEnd of process. On {len(list_sessions)} sessions, {len(sessions_fail)} fails.")
    if (len(sessions_fail)):
        [print("\t* " + session_name) for session_name in sessions_fail]  



if __name__ == "__main__":
    opt = parse_args()
    if opt.is_seatizen_session:
        main_seatizen(opt)
    else:
        main_raster(opt)
