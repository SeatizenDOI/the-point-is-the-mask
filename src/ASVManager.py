import rasterio
import numpy as np
from pathlib import Path
from collections import defaultdict

from .PathManager import PathManager
from .ConfigParser import ConfigParser
from .utils.zenodo_downloader import download_manager_without_token, get_version_from_session_name

class ASVManager:

    def __init__(self, cp: ConfigParser, pm: PathManager) -> None:
        
        self.cp = cp
        self.pm = pm
    
        self.class_quantiles = self.setup()
        self.class_mapping = {}

    
    def setup(self) -> dict:
        
        print("\n\n------ [ASV - Download and get class quantiles] ------\n")
        # Download sessions
        for session in self.cp.asv_sessions:
            session_path = Path(self.pm.asv_sessions_folder, session)
            self.setup_session_asv(session_path)

            session_path_ia = Path(session_path, "PROCESSED_DATA", "IA")
            if not session_path_ia.exists() or not session_path_ia.is_dir() or len(list(session_path_ia.iterdir())) == 0: 
                print(f"Cannot find raster for session {session_path_ia}")
                return
            
            class_probability_values = defaultdict(list)
            for raster in session_path_ia.iterdir():
                if raster.suffix.lower() not in [".tif"]: continue
                
                class_name = raster.name.replace("_raster.tif", "").replace(f"{session_path.name}_", "")
                if class_name not in self.cp.list_label_asv: continue
                
                with rasterio.open(raster) as src:
                    data = src.read(1).astype(np.float32)
                    data = data[~np.isnan(data)]
                    class_probability_values[class_name].extend(data.flatten())
        
        class_quantiles = {}
        for class_name, values in class_probability_values.items():
            values = np.array(values)
            q5, q95 = np.percentile(values, [1, 99])
            class_quantiles[class_name] = (q5, q95)
        
        print(f"\nClass quantiles are: {class_quantiles}")
        return class_quantiles


    def create_coarse_annotations(self) -> None:
        
        print("\n\n------ [ASV - Create coarse annotations] ------\n")

        for session_name in self.cp.asv_sessions:
            
            classification_raster = Path(self.pm.asv_coarse_folder, f"{session_name}_classification.tif")
            if classification_raster.exists() and classification_raster.is_file():
                print(f"{session_name}: Don't perform coarse annotation. Raster already exists")

                if len(self.class_mapping) == 0:
                    # Reading the tags to get min and max labels 
                    with rasterio.open(classification_raster) as src:
                        self.class_mapping = {k: v for k, v in src.tags().items() if k.isdigit()}
                
                continue

            session_path_ia = Path(self.pm.asv_sessions_folder, session_name, "PROCESSED_DATA", "IA")
            raster_files, class_names = [], []
            for raster in sorted(list(session_path_ia.iterdir())):
                if raster.suffix.lower() not in [".tif"]: continue
                
                class_name = raster.name.replace("_raster.tif", "").replace(f"{session_name}_", "")
                if class_name not in self.cp.list_label_asv: continue

                class_names.append(class_name)
                raster_files.append(raster)

            src_files = [rasterio.open(raster) for raster in raster_files]
            raster_data = np.stack([src.read(1).astype(np.float32) for src in src_files], axis=0)

            for i, class_name in enumerate(class_names):
                if class_name in self.class_quantiles:
                    q5, q95 = self.class_quantiles[class_name]
                    band = raster_data[i]
                    valid_mask = ~np.isnan(band)
                    stretched = np.empty_like(band)
                    stretched[:] = np.nan
                    stretched[valid_mask] = (band[valid_mask] - q5) / (q95 - q5 + 1e-6)
                    raster_data[i] = np.clip(stretched, 0, 1)


            max_prob = np.nanmax(raster_data, axis=0)
            classification_map = np.where(np.isnan(max_prob), 255, np.argmax(raster_data, axis=0) + 1).astype(np.uint8)

            out_meta = src_files[0].meta.copy()
            out_meta.update({"count": 1, "dtype": classification_map.dtype, "nodata": 255})

            class_mapping = {str(i + 1): class_names[i] for i in range(len(class_names))}
            if len(self.class_mapping) == 0:
                self.class_mapping = class_mapping

            with rasterio.open(classification_raster, "w", **out_meta) as dest:
                dest.write(classification_map, 1)
                dest.update_tags(**class_mapping)

            print(f"Saved classified raster: {classification_raster}")


    def setup_session_asv(self, session_path: Path) -> None:

        path_ia_session = Path(session_path, "PROCESSED_DATA", "IA")

        if path_ia_session.exists() and len(list(path_ia_session.iterdir())) != 0:
            print(f"Don't download the session {session_path.name}, raster already exists")
            return
        
        version_json = get_version_from_session_name(session_path.name)
        if version_json == {} or "files" not in version_json:
            raise FileNotFoundError(f"Version not found for {session_path.name}")
        
        list_files = [d for d in version_json["files"] if d["key"] in ["PROCESSED_DATA_IA.zip"]]

        # Continue if no files to download due to access_right not open.
        if len(list_files) == 0 and version_json["metadata"]["access_right"] != "open":
            print("[WARNING] No files to download, version is not open.")
            return
            
        # In case we get a conceptrecid from the user, get doi
        doi = version_json["id"]

        download_manager_without_token(list_files, session_path, doi)
    

    def get_classes_mapping(self) -> dict:
        return self.class_mapping