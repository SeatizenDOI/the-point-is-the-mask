import enum
import torch
import pandas as pd
from pathlib import Path
from argparse import Namespace

class Sources(enum.Enum):
    CSV_SESSION = 0
    FOLDER = 1
    SESSION = 2

def get_mode_from_opt(opt: Namespace) -> Sources | None:
    """ Retrieve mode from input option """
    mode = None

    if opt.enable_csv: 
        mode = Sources.CSV_SESSION
    elif opt.enable_folder: 
        mode = Sources.FOLDER
    elif opt.enable_session: 
        mode = Sources.SESSION

    return mode

def get_src_from_mode(mode: Sources, opt: Namespace) -> Path:
    """ Retrieve src path from mode """
    src = Path()

    if mode == Sources.CSV_SESSION:
        src = Path(opt.path_csv_file)
    elif mode == Sources.FOLDER:
        src = Path(opt.path_folder)
    elif mode == Sources.SESSION:
        src = Path(opt.path_session)

    return src

def get_list_rasters(opt: Namespace) -> list[Path]:
    """ Retrieve list of rasters from input """

    list_sessions: list[Path] = []

    mode = get_mode_from_opt(opt)
    if mode == None: return list_sessions

    src = get_src_from_mode(mode, opt)

    if mode == Sources.SESSION:
        list_sessions = [src]

    elif mode == Sources.FOLDER:
        list_sessions = sorted([s for s in src.iterdir() if s.is_file() and s.suffix.lower() == ".tif"])
    
    elif mode == Sources.CSV_SESSION:
        if src.exists():
            df_ses = pd.read_csv(src)
            list_sessions = [Path(row.root_folder, row.ortho_name) for row in df_ses.itertuples(index=False)]

    return list_sessions


def get_list_sessions(opt: Namespace) -> list[Path]:
    """ Retrieve list of sessions from input """

    list_sessions: list[Path] = []

    mode = get_mode_from_opt(opt)
    if mode == None: return list_sessions

    src = get_src_from_mode(mode, opt)

    if mode == Sources.SESSION:
        list_sessions = [src]

    elif mode == Sources.FOLDER:
        list_sessions = sorted([s for s in src.iterdir() if s.is_dir()])
    
    elif mode == Sources.CSV_SESSION:
        if src.exists():
            df_ses = pd.read_csv(src)
            list_sessions = [Path(row.root_folder, row.session_name) for row in df_ses.itertuples(index=False)]

    return list_sessions

def print_header():

    print("""
    
\t\t ▄▖▌     ▄▖  ▘  ▗   ▄▖    ▄▖▌     ▖  ▖    ▌ 
\t\t ▐ ▛▌█▌  ▙▌▛▌▌▛▌▜▘  ▐ ▛▘  ▐ ▛▌█▌  ▛▖▞▌▀▌▛▘▙▘
\t\t ▐ ▌▌▙▖  ▌ ▙▌▌▌▌▐▖  ▟▖▄▌  ▐ ▌▌▙▖  ▌▝ ▌█▌▄▌▛▖
          

""")
    
def print_gpu_is_used() -> None:
    """ Print banner to show if gpu is used. """
    # Check if GPU available
    if torch.cuda.is_available():
        print("\n###################################################")
        print("Using GPU for training.")
        print("###################################################\n")
    else:
        print("GPU not available, using CPU instead.")