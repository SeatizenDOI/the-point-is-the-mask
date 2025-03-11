import traceback
from argparse import Namespace, ArgumentParser

from src.utils.lib_tools import get_list_rasters
from src.PathManager import PathManager

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="Segmentation Inference", description="Workflow to perform segmentation inference on UAV orthophoto.")

    # Input.
    arg_input = parser.add_mutually_exclusive_group(required=True)
    arg_input.add_argument("-efol", "--enable_folder", action="store_true", help="Work from a folder of raster")
    arg_input.add_argument("-eses", "--enable_session", action="store_true", help="Work with one raster")
    arg_input.add_argument("-ecsv", "--enable_csv", action="store_true", help="Work from csv")

    # Path of input.
    parser.add_argument("-pfol", "--path_folder", default="/media/bioeos/E/drone/serge_ortho_tif", help="Path to folder of session")
    parser.add_argument("-pses", "--path_session", default="/media/bioeos/E/drone/serge_ortho_tif/20231202_REU-TROU-DEAU_UAV-01_01_ortho.tif", help="Path to the session")
    parser.add_argument("-pcsv", "--path_csv_file", default=None, help="Path to the csv file")

    # Model arguments.
    parser.add_argument("-psm", "--path_segmentation_model", default="segmentation_model/checkpoint-3807/", help="Path to semgentation model, currently only in local.") # TODO add remote parse
     
    
    # Output.
    parser.add_argument("-po" , "--path_output", default="./output2", help="Path of output")

    # Optional arguments.
    parser.add_argument("-is", "--index_start", default="0", help="Choose from which index to start")
    parser.add_argument("-c", "--clean", action="store_true", help="Delete all previous file")

    return parser.parse_args()

def main(opt: Namespace) -> None:
    
    list_rasters = get_list_rasters(opt)
    index_start = int(opt.index_start) if opt.index_start.isnumeric() and int(opt.index_start) < len(list_rasters) else 0
    list_rasters = list_rasters[index_start:]

    rasters_fail = []
    for raster_path in list_rasters:
        # Filter all files who are not rasters files
        if not raster_path.is_file() or raster_path.suffix != ".tif": continue
        
        path_manager = PathManager(opt.path_output, raster_path.stem)

        # Clean if needed
        path_manager.clean() if opt.clean else path_manager.create_path()

        try:
            pass



        except Exception as e:
            print(traceback.format_exc(), end="\n\n")
            rasters_fail.append(raster_path.name)
        finally:
            path_manager.disk_optimize()
    
    # Stat
    print(f"\nEnd of process. On {len(list_rasters)} sessions, {len(rasters_fail)} fails.")
    if (len(rasters_fail)):
        [print("\t* " + session_name) for session_name in rasters_fail]  


if __name__ == "__main__":
    opt = parse_args()
    main(opt)