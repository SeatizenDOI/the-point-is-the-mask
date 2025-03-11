import shutil
from pathlib import Path

CROPPED_ORTHO = "cropped_ortho"
CROPPED_ORTHO_IMG = "cropped_ortho_img"
PREDICTIONS_TIFF = "predictions_tiff"
MERGED_PREDICTIONS = "merged_predictions"


class PathManager:


    def __init__(self, output_folder: str, session_name: str):

        self.output_folder = Path(output_folder)
        self.cropped_ortho_folder = Path(output_folder, CROPPED_ORTHO, session_name)
        self.cropped_ortho_img_folder = Path(output_folder, CROPPED_ORTHO_IMG, session_name)
        self.predictions_tiff_folder = Path(output_folder, PREDICTIONS_TIFF, session_name)
        self.merged_predictions_folder = Path(output_folder, MERGED_PREDICTIONS)
    


    def clean(self):
        """ Remvoe previous intermediate files and create path. """
        self.disk_optimize()
        self.create_path()

    def create_path(self) -> None:
        """ Create all path associate to the session. """
        self.cropped_ortho_folder.mkdir(exist_ok=True, parents=True)
        self.cropped_ortho_img_folder.mkdir(exist_ok=True, parents=True)
        self.predictions_tiff_folder.mkdir(exist_ok=True, parents=True)
        self.merged_predictions_folder.mkdir(exist_ok=True, parents=True)


    def disk_optimize(self) -> None:
        """ Remove all intermediate files"""

        if self.cropped_ortho_folder.exists():
            shutil.rmtree(self.cropped_ortho_folder)
        if self.cropped_ortho_img_folder.exists():
            shutil.rmtree(self.cropped_ortho_img_folder)
        if self.predictions_tiff_folder.exists():
            shutil.rmtree(self.predictions_tiff_folder)
