import shutil
from pathlib import Path

from .ConfigParser import ConfigParser

class PathManager:

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

        self.asv_folder = Path(self.output_path, "asv")
        self.uav_folder = Path(self.output_path, "uav")

        self.asv_sessions_folder = Path(self.asv_folder, "sessions")
        self.asv_coarse_folder = Path(self.asv_folder, "coarse")
        self.uav_sessions_folder = Path(self.uav_folder, "sessions")

        ## Coarse path.
        self.tiles_coarse_folder = Path(self.output_path, "tiles_coarse")
        self.coarse_cropped_ortho_tif_folder = Path(self.tiles_coarse_folder, "cropped_ortho_tif")
        self.coarse_upsampled_annotation_tif_folder = Path(self.tiles_coarse_folder, "upsampled_annotation_tif")

        self.coarse_train_folder = Path(self.tiles_coarse_folder, "train")
        self.coarse_train_images_folder = Path(self.coarse_train_folder, "images")
        self.coarse_train_annotation_folder = Path(self.coarse_train_folder, "annotations")
        
        self.coarse_test_folder = Path(self.tiles_coarse_folder, "test")
        self.coarse_test_images_folder = Path(self.coarse_test_folder, "images")
        self.coarse_test_annotation_folder = Path(self.coarse_test_folder, "annotations")

        # Refine path.
        self.tiles_refine_folder = Path(self.output_path, "tiles_refine")
        self.refine_cropped_ortho_tif_folder = Path(self.tiles_refine_folder, "cropped_ortho_tif")
        self.refine_annotation_tif_folder = Path(self.tiles_refine_folder, "annotation_tif")

        self.refine_train_folder = Path(self.tiles_refine_folder, "train")
        self.refine_train_images_folder = Path(self.refine_train_folder, "images")
        self.refine_train_annotation_folder = Path(self.refine_train_folder, "annotations")
        
        self.refine_test_folder = Path(self.tiles_refine_folder, "test")
        self.refine_test_images_folder = Path(self.refine_test_folder, "images")
        self.refine_test_annotation_folder = Path(self.refine_test_folder, "annotations")

        self.uav_csv = Path(self.uav_folder, "uav_sessions.csv")
        self.uav_prediction_refine_raster_folder = Path(self.output_path, "final_predictions_raster")



    def setup(self, cp: ConfigParser) -> None:
        print("------ [CLEANING] ------")
        if cp.clean_asv_session() and self.asv_sessions_folder.exists():
            print(f"* Delete {self.asv_sessions_folder}")
            shutil.rmtree(self.asv_sessions_folder)

        if cp.clean_asv_coarse() and self.asv_coarse_folder.exists():
            print(f"* Delete {self.asv_coarse_folder}")
            shutil.rmtree(self.asv_coarse_folder)

        if cp.clean_uav_session() and self.uav_sessions_folder.exists():
            print(f"* Delete {self.uav_sessions_folder}")
            shutil.rmtree(self.uav_sessions_folder)

        if cp.clean_coarse_cropped_ortho_tif() and self.coarse_cropped_ortho_tif_folder.exists():
            print(f"* Delete {self.coarse_cropped_ortho_tif_folder}")
            shutil.rmtree(self.coarse_cropped_ortho_tif_folder)

        if cp.clean_coarse_upscaling_annotation_tif() and self.coarse_upsampled_annotation_tif_folder.exists():
            print(f"* Delete {self.coarse_upsampled_annotation_tif_folder}")
            shutil.rmtree(self.coarse_upsampled_annotation_tif_folder)

        if cp.clean_coarse_train() and self.coarse_train_folder.exists():
            print(f"* Delete {self.coarse_train_folder}")
            shutil.rmtree(self.coarse_train_folder)

        if cp.clean_coarse_test() and self.coarse_test_folder.exists():
            print(f"* Delete {self.coarse_test_folder}")
            shutil.rmtree(self.coarse_test_folder)

        if cp.clean_refine_cropped_ortho_tif() and self.refine_cropped_ortho_tif_folder.exists():
            print(f"* Delete {self.refine_cropped_ortho_tif_folder}")
            shutil.rmtree(self.refine_cropped_ortho_tif_folder)

        if cp.clean_refine_upscaling_annotation_tif() and self.refine_annotation_tif_folder.exists():
            print(f"* Delete {self.refine_annotation_tif_folder}")
            shutil.rmtree(self.refine_annotation_tif_folder)

        if cp.clean_refine_train() and self.refine_train_folder.exists():
            print(f"* Delete {self.refine_train_folder}")
            shutil.rmtree(self.refine_train_folder)

        if cp.clean_refine_test() and self.refine_test_folder.exists():
            print(f"* Delete {self.refine_test_folder}")
            shutil.rmtree(self.refine_test_folder)

        print(f"* Create all subfolders.")
        self.asv_coarse_folder.mkdir(exist_ok=True, parents=True)
        self.asv_sessions_folder.mkdir(exist_ok=True, parents=True)
        self.uav_sessions_folder.mkdir(exist_ok=True, parents=True)
        self.coarse_cropped_ortho_tif_folder.mkdir(exist_ok=True, parents=True)
        self.coarse_upsampled_annotation_tif_folder.mkdir(exist_ok=True, parents=True)
        self.coarse_train_images_folder.mkdir(exist_ok=True, parents=True)
        self.coarse_train_annotation_folder.mkdir(exist_ok=True, parents=True)
        self.coarse_test_images_folder.mkdir(exist_ok=True, parents=True)
        self.coarse_test_annotation_folder.mkdir(exist_ok=True, parents=True)
        self.refine_cropped_ortho_tif_folder.mkdir(exist_ok=True, parents=True)
        self.refine_annotation_tif_folder.mkdir(exist_ok=True, parents=True)
        self.refine_train_images_folder.mkdir(exist_ok=True, parents=True)
        self.refine_train_annotation_folder.mkdir(exist_ok=True, parents=True)
        self.refine_test_images_folder.mkdir(exist_ok=True, parents=True)
        self.refine_test_annotation_folder.mkdir(exist_ok=True, parents=True)
