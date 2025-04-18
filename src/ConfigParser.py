import json
from pathlib import Path
from argparse import Namespace


class ConfigParser:

    def __init__(self, opt: Namespace) -> None:
        self.opt = opt

        self.config_json = self.load_config_json()

    def load_config_json(self) -> dict:

        config_path = Path(self.opt.config_path)
        if not config_path.exists() or not config_path.is_file():
            raise NameError(f"Cannot find config path at {config_path}")
        
        config_data = {}
        with open(config_path) as f:
            config_data = json.load(f)

        return config_data

    ## Getter part.

    def get_asv_sessions(self) -> list:
        return self.config_json.get("list_asv_sessions", [])

    def get_uav_sessions(self) -> list:
            return self.config_json.get("list_uav_sessions", [])

    def get_output_path(self) -> Path:
        return Path(self.config_json.get("output_path", None))

    def get_list_label_asv(self) -> list:
        return self.config_json.get("list_labels_asv_predictions", [])


    ## Clean.

    def clean_asv_session(self) -> bool:
        clean_dict = self.config_json.get("clean", {})
        if clean_dict == {}: return False

        return bool(clean_dict.get("asv_session", False))

    def clean_asv_coarse(self) -> bool:
        clean_dict = self.config_json.get("clean", {})
        if clean_dict == {}: return False

        return bool(clean_dict.get("asv_coarse", False))
    
    def clean_uav_session(self) -> bool:
        clean_dict = self.config_json.get("clean", {})
        if clean_dict == {}: return False

        return bool(clean_dict.get("uav_session", False))