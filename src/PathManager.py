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


    def setup(self, cp: ConfigParser) -> None:
        print("------ [CLEANING] ------")
        if cp.clean_asv_session() and self.asv_sessions_folder.exists():
            print(f"\t * Delete {self.asv_sessions_folder}")
            shutil.rmtree(self.asv_sessions_folder)

        if cp.clean_asv_coarse() and self.asv_coarse_folder.exists():
            print(f"\t * Delete {self.asv_coarse_folder}")
            shutil.rmtree(self.asv_coarse_folder)

        if cp.clean_uav_session() and self.uav_sessions_folder.exists():
            print(f"\t * Delete {self.uav_sessions_folder}")
            shutil.rmtree(self.uav_sessions_folder)

        print(f"\t * Create all subfolders.")
        self.asv_coarse_folder.mkdir(exist_ok=True, parents=True)
        self.asv_sessions_folder.mkdir(exist_ok=True, parents=True)
        self.uav_sessions_folder.mkdir(exist_ok=True, parents=True)