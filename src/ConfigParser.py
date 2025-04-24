import json
from pathlib import Path
from argparse import Namespace

CLEAN_PARAMETERS = "clean"
TILES_PARAMETERS = "tiles"
MODEL_PARAMETERS = "models"
class ConfigParser:

    def __init__(self, opt: Namespace) -> None:
        self.opt = opt

        self.config_json = self.load_config_json()
        self.env_json = self.load_env_file()
    
        self.tiles_dict, self.clean_dict, self.model_dict = {}, {}, {}
        self.verify_basic_header_exists()

    def load_config_json(self) -> dict:

        config_path = Path(self.opt.config_path)
        if not config_path.exists() or not config_path.is_file():
            raise FileNotFoundError(f"Cannot find config path at {config_path}")
        
        config_data = {}
        with open(config_path) as f:
            config_data = json.load(f)

        return config_data
    
    def load_env_file(self) -> dict:

        env_path = Path(self.opt.env_path)
        if not env_path.exists() or not env_path.is_file():
            raise FileNotFoundError(f"Cannot find config path at {env_path}")
        
        env_data = {}
        with open(env_path, "r") as file:
            for row in file:
                k, v = row.replace("\n", "").split("=")
                env_data[k] = v
        
        return env_data

    
    def verify_basic_header_exists(self) -> None:
        self.clean_dict = self.config_json.get(CLEAN_PARAMETERS, {})
        if self.clean_dict == {}: raise NameError(f"Cannot find {CLEAN_PARAMETERS} in config json")

        self.tiles_dict = self.config_json.get(TILES_PARAMETERS, {})
        if self.tiles_dict == {}: raise NameError(f"Cannot find {TILES_PARAMETERS} in config json")

        self.model_dict = self.config_json.get(MODEL_PARAMETERS, {})
        if self.model_dict == {}: raise NameError(f"Cannot find {MODEL_PARAMETERS} in config json")


    ## Getter part.

    def get_asv_sessions(self) -> list:
        return self.config_json.get("list_asv_sessions", [])

    def get_uav_sessions(self) -> list:
            return self.config_json.get("list_uav_sessions", [])

    def get_output_path(self) -> Path:
        return Path(self.config_json.get("output_path", None))

    def get_list_label_asv(self) -> list:
        return self.config_json.get("list_labels_asv_predictions", [])
    
    def get_drone_zone_polygon_path(self) -> list:
        return self.config_json.get("drone_test_zone_polygon_path", [])
    
    def get_hugging_face_token(self) -> str:
        return self.env_json.get("HUGGINGFACE_TOKEN", "")
    

    ## Tiles.
    def get_tile_size(self) -> int:
        return int(self.tiles_dict.get("tile_size", 0))

    def get_horizontal_overlap(self) -> int:
        return int(self.tiles_dict.get("horizontal_overlap", 0))

    def get_vertical_overlap(self) -> int:
        return int(self.tiles_dict.get("vertical_overlap", 0))

    def get_horizontal_step(self) -> int:
        return int(self.get_tile_size() * (1 - self.get_horizontal_overlap()))

    def get_vertical_step(self) -> int:
        return int(self.get_tile_size() * (1 - self.get_vertical_overlap()))
    
    def use_color_correction(self) -> bool:
        return bool(self.tiles_dict.get("with_color_correction", False))

    ## Clean.
    def clean_asv_session(self) -> bool:
        return bool(self.clean_dict.get("asv_session", False))

    def clean_asv_coarse(self) -> bool:
        return bool(self.clean_dict.get("asv_coarse", False))
    
    def clean_uav_session(self) -> bool:
        return bool(self.clean_dict.get("uav_session", False))

    def clean_cropped_ortho_tif(self) -> bool:
        return bool(self.clean_dict.get("cropped_ortho_tif", False))    
    
    def clean_upscaling_annotation_tif(self) -> bool:
        return bool(self.clean_dict.get("upscaling_annotation_tif", False))
    
    def clean_train(self) -> bool:
        return bool(self.clean_dict.get("train", False))
    
    def clean_test(self) -> bool:
        return bool(self.clean_dict.get("test", False))
    
    ## Models.
    @property
    def model_name(self) -> str:
        return str(self.model_dict.get("model_name", ""))
    
    @property
    def epochs(self) -> int:
        return int(self.model_dict.get("epochs", 0))
    
    @property
    def batch_size(self) -> int:
        return int(self.model_dict.get("batch_size", 0))

    @property
    def initial_learning_rate(self) -> float:
        return float(self.model_dict.get("initial_learning_rate", 0))
    
    @property
    def weight_decay(self) -> float:
        return float(self.model_dict.get("weight_decay", 0))
    
    @property
    def factor_lr_scheduler(self) -> float:
        return float(self.model_dict.get("factor_lr_scheduler", 0))
    
    @property
    def patience_lr_scheduler(self) -> int:
        return int(self.model_dict.get("patience_lr_scheduler", 0))

    @property
    def early_stopping_patience(self) -> int:
        return int(self.model_dict.get("early_stopping_patience", 0))
    
    @property
    def path_output_dir(self) -> str:
        return self.model_dict.get("path_output_dir", "")