import traceback
from datetime import datetime
from argparse import Namespace, ArgumentParser

from src.utils.lib_tools import get_list_rasters
from src.inference.PathRasterManager import PathRasterManager
from src.inference.TileManager import TileManager
from src.inference.ModelManager import ModelManager
from src.inference.MosaicManager import MosaicManager

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="Segmentation Inference", description="Workflow to perform segmentation inference on UAV orthophoto.")

    # Input.
    arg_input = parser.add_mutually_exclusive_group(required=True)
    arg_input.add_argument("-efol", "--enable_folder", action="store_true", help="Work from a folder of raster")
    arg_input.add_argument("-eses", "--enable_session", action="store_true", help="Work with one raster")
    arg_input.add_argument("-ecsv", "--enable_csv", action="store_true", help="Work from csv")

    # Path of input.
    parser.add_argument("-pfol", "--path_folder", default="input", help="Path to folder of session")
    parser.add_argument("-pses", "--path_session", default="input/20231202_REU-TROU-DEAU_UAV-01_01_ortho.tif", help="Path to the session")
    parser.add_argument("-pcsv", "--path_csv_file", default=None, help="Path to the csv file")

    # Model arguments.
    parser.add_argument("-psm", "--path_segmentation_model", default="segmentation_model/quantile99-segmentation_model-ce0-dice1/checkpoint-5358", help="Path to semgentation model, currently only in local.") # TODO add remote parse
    parser.add_argument("-pgeo", "--path_geojson", default="config/emprise_lagoon.geojson", help="Path to geojson to crop ortho inside area.")
    parser.add_argument("-ho", "--horizontal_overlap", type=float, default=0.5, help="Horizontal overlap between tiles.")
    parser.add_argument("-vo", "--vertical_overlap", type=float, default=0.5, help="Vertical overlap between tiles.")
    parser.add_argument("-ts", "--tile_size", type=int, default=512, help="Split Orthophoto into tiles.")
    parser.add_argument("--geojson_crs", default="EPSG:4326", help="Defaulting to WGS84 (likely for GeoJSON)")
    parser.add_argument("-ucc", "--underwater_color_correction", action="store_true", help="Perform color corection on water.")

    # Output.
    parser.add_argument("-po" , "--path_output", default="./output", help="Path of output")

    # Optional arguments.
    parser.add_argument("-is", "--index_start", default="0", help="Choose from which index to start")
    parser.add_argument("-c", "--clean", action="store_true", help="Delete all previous file")

    return parser.parse_args()

def main(opt: Namespace) -> None:

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
            
            mosaic_manager = MosaicManager(path_manager)
            mosaic_manager.build_raster()
            
        except Exception as e:
            print(traceback.format_exc(), end="\n\n")
            rasters_fail.append(raster_path.name)
        finally:
            if opt.clean:
                path_manager.disk_optimize() 

        print(f"\n*\t Running time {datetime.now() - t_start}")

    # Stat
    print(f"\nEnd of process. On {len(list_rasters)} sessions, {len(rasters_fail)} fails.")
    if (len(rasters_fail)):
        [print("\t* " + session_name) for session_name in rasters_fail]  


if __name__ == "__main__":
    opt = parse_args()
    print(opt)
    # main(opt)