import shutil
from pathlib import Path

CROPPED_ORTHO = "cropped_ortho"
CROPPED_ORTHO_IMG = "cropped_ortho_img"
PREDICTIONS_TIFF = "predictions_tiff"
PREDICTIONS_PNG = "predictions_png"
MERGED_PREDICTIONS = "final_predictions_raster"


class PathRasterManager:

    def __init__(self, output_folder: str, raster_path: Path):

        self.raster_path = raster_path
        self.output_folder = Path(output_folder)
        self.tmp_folder = Path(output_folder, "tmp")
        self.cropped_ortho_folder = Path(self.tmp_folder, CROPPED_ORTHO, raster_path.stem)
        self.cropped_ortho_img_folder = Path(self.tmp_folder, CROPPED_ORTHO_IMG, raster_path.stem)
        self.predictions_tiff_folder = Path(self.tmp_folder, PREDICTIONS_TIFF, raster_path.stem)
        self.predictions_png_folder = Path(self.tmp_folder, PREDICTIONS_PNG, raster_path.stem)
        self.merged_predictions_folder = Path(output_folder, MERGED_PREDICTIONS)
        self.final_merged_tiff_file = Path(self.merged_predictions_folder, f"{raster_path.stem}_merged_predictions.tif")

    def is_empty_cropped_folder(self) -> bool:
        return len(list(self.cropped_ortho_folder.iterdir())) == 0

    def is_empty_cropped_img_folder(self) -> bool:
        return len(list(self.cropped_ortho_img_folder.iterdir())) == 0

    def is_empty_predictions_tiff_folder(self) -> bool:
        return len(list(self.predictions_tiff_folder.iterdir())) == 0

    def clean(self):
        """ Remvoe previous intermediate files and create path. """
        self.disk_optimize()
        self.create_path()

    def create_path(self) -> None:
        """ Create all path associate to the session. """
        print("*\t Create sub folder. ")
        self.cropped_ortho_folder.mkdir(exist_ok=True, parents=True)
        self.cropped_ortho_img_folder.mkdir(exist_ok=True, parents=True)
        self.predictions_tiff_folder.mkdir(exist_ok=True, parents=True)
        self.predictions_png_folder.mkdir(exist_ok=True, parents=True)
        self.merged_predictions_folder.mkdir(exist_ok=True, parents=True)

    def disk_optimize(self) -> None:
        """ Remove all intermediate files"""
        print("*\t Remove all folder if exists. ")
        if self.tmp_folder.exists():
            shutil.rmtree(self.tmp_folder)