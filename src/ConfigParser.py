import json
from pathlib import Path
from argparse import Namespace

GLOBAL_PARAM = "global"
CLEAN_PARAM = "clean"
SETUP_PARAM = "setup"
TILES_PARAM = "tiles"
TRAIN_PARAM = "train"
MODEL_PARAM = "parameters"
INFERENCE_PARAM = "inference"
class ConfigParser:

    def __init__(self, opt: Namespace) -> None:
        self.opt = opt

        self.config_json = self.load_config_json()
        self.env_json = self.load_env_file()
    
        self.tiles_dict, self.clean_dict, self.model_dict = {}, {}, {}
        self.global_dict, self.setup_dict, self.train_dict = {}, {}, {}
        self.inference_dict = {}

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

        self.global_dict = self.config_json.get(GLOBAL_PARAM, {})
        if self.global_dict == {}: raise NameError(f"Cannot find {GLOBAL_PARAM} in config json")

        self.clean_dict = self.global_dict.get(CLEAN_PARAM, {})
        if self.clean_dict == {}: raise NameError(f"Cannot find {CLEAN_PARAM} in config json")

        self.setup_dict = self.config_json.get(SETUP_PARAM, {})
        if self.setup_dict == {}: raise NameError(f"Cannot find {SETUP_PARAM} in config json")

        self.tiles_dict = self.setup_dict.get(TILES_PARAM, {})
        if self.tiles_dict == {}: raise NameError(f"Cannot find {TILES_PARAM} in config json")

        self.train_dict = self.config_json.get(TRAIN_PARAM, {})
        if self.train_dict == {}: raise NameError(f"Cannot find {TRAIN_PARAM} in config json")

        self.model_dict = self.train_dict.get(MODEL_PARAM, {})
        if self.model_dict == {}: raise NameError(f"Cannot find {MODEL_PARAM} in config json")

        self.inference_dict = self.config_json.get(INFERENCE_PARAM, {})
        if self.inference_dict == {}: raise NameError(f"Cannot find {INFERENCE_PARAM} in config json")



    ## Getter part.
    @property
    def asv_sessions(self) -> list:
        return self.setup_dict.get("list_asv_sessions", [])

    @property
    def uav_sessions(self) -> list:
            return self.setup_dict.get("list_uav_sessions", [])

    @property
    def output_path(self) -> Path:
        return Path(self.global_dict.get("output_path", None))

    @property
    def list_label_asv(self) -> list:
        return self.setup_dict.get("list_labels_asv_predictions", [])
    
    @property
    def drone_zone_polygon_path(self) -> list:
        return self.setup_dict.get("drone_test_zone_polygon_path", [])
    
    @property
    def hugging_face_token(self) -> str:
        return self.env_json.get("HUGGINGFACE_TOKEN", "")
    

    ## Tiles.
    @property
    def tile_size(self) -> int:
        return int(self.tiles_dict.get("tile_size", 0))

    @property
    def horizontal_overlap(self) -> int:
        return int(self.tiles_dict.get("horizontal_overlap", 0))
    
    @property
    def vertical_overlap(self) -> int:
        return int(self.tiles_dict.get("vertical_overlap", 0))

    @property
    def horizontal_step(self) -> int:
        return int(self.tile_size * (1 - self.horizontal_overlap))

    @property
    def vertical_step(self) -> int:
        return int(self.tile_size * (1 - self.vertical_overlap))
    
    @property
    def use_color_correction(self) -> bool:
        return bool(self.tiles_dict.get("with_color_correction", False))

    ## Clean.
    def clean_asv_session(self) -> bool:
        return bool(self.clean_dict.get("asv_session", False))

    def clean_asv_coarse(self) -> bool:
        return bool(self.clean_dict.get("asv_coarse", False))
    
    def clean_uav_session(self) -> bool:
        return bool(self.clean_dict.get("uav_session", False))

    def clean_coarse_cropped_ortho_tif(self) -> bool:
        return bool(self.clean_dict.get("coarse_cropped_ortho_tif", False))    
    
    def clean_coarse_upscaling_annotation_tif(self) -> bool:
        return bool(self.clean_dict.get("coarse_upscaling_annotation_tif", False))
    
    def clean_coarse_train(self) -> bool:
        return bool(self.clean_dict.get("coarse_train", False))
    
    def clean_coarse_test(self) -> bool:
        return bool(self.clean_dict.get("coarse_test", False))

    def clean_refine_cropped_ortho_tif(self) -> bool:
        return bool(self.clean_dict.get("refine_cropped_ortho_tif", False))    
    
    def clean_refine_upscaling_annotation_tif(self) -> bool:
        return bool(self.clean_dict.get("refine_upscaling_annotation_tif", False))
    
    def clean_refine_train(self) -> bool:
        return bool(self.clean_dict.get("refine_train", False))
    
    def clean_refine_test(self) -> bool:
        return bool(self.clean_dict.get("refine_test", False))

    def clean_eval(self) -> bool:
        return bool(self.clean_dict.get("eval", False))
    
    ## Models.
    @property
    def base_model_name(self) -> str:
        return str(self.train_dict.get("base_model", ""))
    
    @property
    def model_name(self) -> str:
        return str(self.train_dict.get("model_name", ""))
    
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
    def path_models_checkpoints(self) -> str:
        return self.train_dict.get("path_output_dir", "")
    
    @property
    def resume_coarse_training(self) -> str | None:
        t = self.train_dict.get("coarse_training", None)
        if t == None: return None

        return t.get("resume_from", "")

    @property
    def model_path_coarse(self) -> Path | None:
        t = self.train_dict.get("coarse_training", None)
        if t == None: return None
        p = t.get("model_path", None)
        return Path(p) if p != None else None

    @property
    def resume_refine_training(self) -> str | None:
        t = self.train_dict.get("refine_training", None)
        if t == None: return None

        return t.get("resume_from", "")

    @property
    def upload_on_huggingface(self) -> bool:
        t = self.train_dict.get("refine_training", False)
        if t == False: return False

        return t.get("upload_on_huggingface", False)

    @property
    def model_path_refine(self) -> Path | None:
        t = self.train_dict.get("refine_training", None)
        if t == None: return None
        p = t.get("model_path", None)
        return Path(p) if p != None else None
    
    @property
    def with_sam_refiner(self) -> bool:
        return bool(self.inference_dict.get("with_sam_refiner", True))
    
    @property
    def path_sam_model(self) -> str:
        return str(self.inference_dict.get("path_sam_model", ""))

    @property
    def list_geojson_to_keep_inference(self) -> list:
        return self.inference_dict.get("list_geojson_to_keep", [])