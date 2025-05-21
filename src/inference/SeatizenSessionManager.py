import shutil
from pathlib import Path

class SeatizenSessionManager:

    def __init__(self, session_path: Path) -> None:
        
        self.session_path = session_path

    
    def get_orthophoto_path(self) -> Path:

        orthophoto_path = Path(self.session_path, "PROCESSED_DATA", "PHOTOGRAMMETRY", "odm_orthophoto", "odm_orthophoto.tif")

        if not orthophoto_path.exists():
            raise FileNotFoundError(f"{orthophoto_path} not found.")
        
        return orthophoto_path
    
    def move_prediction_raster(self, prediction_raster: Path, model_name: str) -> None:

        IA_path = Path(self.session_path, "PROCESSED_DATA", "IA")
        IA_path.mkdir(exist_ok=True, parents=True)

        new_prediction_raster_path = Path(IA_path, f"{self.session_path.name}_{model_name}_ortho_predictions.tif")
        shutil.move(prediction_raster, new_prediction_raster_path)