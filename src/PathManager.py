import shutil
from pathlib import Path

from .ConfigParser import ConfigParser

class PathManager:

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

        self.asv_folder = Path(self.output_path, "asv")
        self.uav_folder = Path(self.output_path, "uav")
        self.tiles_folder = Path(self.output_path, "tiles")
        self.asv_sessions_folder = Path(self.asv_folder, "sessions")
        self.asv_coarse_folder = Path(self.asv_folder, "coarse")
        self.uav_sessions_folder = Path(self.uav_folder, "sessions")
        self.cropped_ortho_tif_folder = Path(self.tiles_folder, "cropped_ortho_tif")
        self.upsampled_annotation_tif_folder = Path(self.tiles_folder, "upsampled_annotation_tif")

        self.train_folder = Path(self.tiles_folder, "train")
        self.train_images_folder = Path(self.train_folder, "images")
        self.train_annotation_folder = Path(self.train_folder, "annotations")
        
        self.test_folder = Path(self.tiles_folder, "test")
        self.test_images_folder = Path(self.test_folder, "images")
        self.test_annotation_folder = Path(self.test_folder, "annotations")


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

        if cp.clean_cropped_ortho_tif() and self.cropped_ortho_tif_folder.exists():
            print(f"* Delete {self.cropped_ortho_tif_folder}")
            shutil.rmtree(self.cropped_ortho_tif_folder)

        if cp.clean_upscaling_annotation_tif() and self.upsampled_annotation_tif_folder.exists():
            print(f"* Delete {self.upsampled_annotation_tif_folder}")
            shutil.rmtree(self.upsampled_annotation_tif_folder)

        if cp.clean_train() and self.train_folder.exists():
            print(f"* Delete {self.train_folder}")
            shutil.rmtree(self.train_folder)

        if cp.clean_test() and self.test_folder.exists():
            print(f"* Delete {self.test_folder}")
            shutil.rmtree(self.test_folder)

        print(f"* Create all subfolders.")
        self.asv_coarse_folder.mkdir(exist_ok=True, parents=True)
        self.asv_sessions_folder.mkdir(exist_ok=True, parents=True)
        self.uav_sessions_folder.mkdir(exist_ok=True, parents=True)
        self.cropped_ortho_tif_folder.mkdir(exist_ok=True, parents=True)
        self.upsampled_annotation_tif_folder.mkdir(exist_ok=True, parents=True)
        self.train_images_folder.mkdir(exist_ok=True, parents=True)
        self.train_annotation_folder.mkdir(exist_ok=True, parents=True)
        self.test_images_folder.mkdir(exist_ok=True, parents=True)
        self.test_annotation_folder.mkdir(exist_ok=True, parents=True)
