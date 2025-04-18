
from argparse import Namespace, ArgumentParser
from pathlib import Path

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.ASVManager import ASVManager
from src.utils.lib_tools import print_header

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="The point is the mask training", description="Workflow to WSSS training on UAV orthophoto using ASV prediction.")

    # Input.
    parser.add_argument("-cp", "--config_path", default="./config.json", help="Path to the config file.")

    

    return parser.parse_args()

def main(opt: Namespace) -> None:

    print_header()
    
    # Initialize parser.
    cp = ConfigParser(opt)

    # Initialize path manager.
    pm = PathManager(cp.get_output_path())
    pm.setup(cp)

    # Create coarse annotations.
    asv_manager = ASVManager(cp, pm)
    asv_manager.create_coarse_annotations()



if __name__ == "__main__":
    opt = parse_args()
    main(opt)