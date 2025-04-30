import rasterio
import pandas as pd
from pathlib import Path

from .PathManager import PathManager
from .ConfigParser import ConfigParser
from .utils.zenodo_downloader import download_manager_without_token, get_version_from_session_name

class UAVManager:

    def __init__(self, cp: ConfigParser, pm: PathManager) -> None:
        
        self.cp = cp
        self.pm = pm

        self.ortho_information, self.ortho_information_by_name = {}, {}
        self.default_crs_uav = None
        self.setup()

    
    def setup(self) -> dict:
        
        print("\n\n------ [UAV - Download photog] ------\n")
        # Download sessions
        for session in self.cp.uav_sessions:
            session_path = Path(self.pm.uav_sessions_folder, session)
            self.setup_session_uav(session_path)

            session_path_photog = Path(session_path, "PROCESSED_DATA", "PHOTOGRAMMETRY")
            if not session_path_photog.exists() or not session_path_photog.is_dir() or len(list(session_path_photog.iterdir())) == 0: 
                print(f"Cannot find raster folder for session {session_path_photog}")
                return
            
            orthophoto_path = Path(session_path_photog, "odm_orthophoto", "odm_orthophoto.tif")
            if not orthophoto_path.exists() or not orthophoto_path.is_file(): 
                print(f"Cannot find raster for session {orthophoto_path}")
                return

            # Create a symlink to the raster to avoid keep odm_orthophoto_name
            ortho_symlink_path = Path(orthophoto_path.parent, f"./{session}_ortho.tif")
            if ortho_symlink_path.is_symlink():
                ortho_symlink_path.unlink()
            ortho_symlink_path.symlink_to("./odm_orthophoto.tif")

            # Extract crs.
            with rasterio.open(ortho_symlink_path) as ortho: 
                ortho_crs = ortho.crs
                ortho_width = ortho.width
                ortho_height = ortho.height

            place_with_country_code = session_path.name.split("_")[1]
            if place_with_country_code not in self.ortho_information:
                self.ortho_information[place_with_country_code] = []
            
            self.ortho_information[place_with_country_code].append((
                ortho_symlink_path,
                ortho_crs,
                ortho_height,
                ortho_width
            ))
            self.ortho_information_by_name[ortho_symlink_path.name] = [
                ortho_symlink_path,
                ortho_crs,
                ortho_height,
                ortho_width
            ]

            self.default_crs_uav = ortho_crs
        
        # Before continue, make sure if all uav orthophoto are in the same CRS.
        nb_unique_crs = len(set([crs for a in list(self.ortho_information.values()) for path, crs, w, h in a]))
        if nb_unique_crs != 1:
            raise ValueError("UAV orthophoto are not in the same CRS. Please consider normalize before.")


    def setup_session_uav(self, session_path: Path) -> None:

        path_photog_session = Path(session_path, "PROCESSED_DATA", "PHOTOGRAMMETRY")

        if path_photog_session.exists() and len(list(path_photog_session.iterdir())) != 0:
            print(f"Don't download the session {session_path.name}, photog folder already exists")
            return
        
        version_json = get_version_from_session_name(session_path.name)
        if version_json == {} or "files" not in version_json:
            raise FileNotFoundError(f"Version not found for {session_path.name}")
        
        list_files = [d for d in version_json["files"] if d["key"] in ["PROCESSED_DATA_PHOTOGRAMMETRY.zip"]]

        # Continue if no files to download due to access_right not open.
        if len(list_files) == 0 and version_json["metadata"]["access_right"] != "open":
            print("[WARNING] No files to download, version is not open.")
            return
            
        # In case we get a conceptrecid from the user, get doi
        doi = version_json["id"]

        download_manager_without_token(list_files, session_path, doi)
    

    def get_orthos_from_place_with_country_code(self, place_with_country_code: str) -> list:
        return self.ortho_information.get(place_with_country_code, [])
    

    def get_default_crs(self) -> str:
        if self.default_crs_uav == None:
            raise ValueError("Cannot get CRS for UAV")

        return self.default_crs_uav
    
    def generate_csv_uav_sessions_for_inference(self) -> None:
        
        orthos = []
        for args in list(self.ortho_information.values()):
            for path, crs, w, h in args:
                path = Path(path)
                orthos.append({"root_folder": path.parent, "ortho_name": path.name})
        
        df_orthos = pd.DataFrame(orthos)
        df_orthos.to_csv(self.pm.uav_csv, index=False)


    def get_ortho_information_from_ortho_name(self, ortho_name: str) -> list:
        return self.ortho_information_by_name.get(ortho_name, [])