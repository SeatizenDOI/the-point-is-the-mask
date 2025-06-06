import time
import shutil
import hashlib
import zipfile
import requests
import traceback
from tqdm import tqdm
from pathlib import Path


ZENODO_LINK_WITHOUT_TOKEN = "https://zenodo.org/api/records"
MAX_RETRY_TO_UPLOAD_DOWNLOAD_FILE = 50


def download_manager_without_token(files: list, session_path: Path, doi: str) -> None:
    """ Manage to download files without token. """
    path_zip_session = Path(session_path, "ZIP")
    path_zip_session.mkdir(exist_ok=True, parents=True)

    for file in files:

        path_tmp_file = Path(path_zip_session, file["key"])
        url = f"{ZENODO_LINK_WITHOUT_TOKEN}/{doi}/files/{file['key']}/content"
        print(f"\nWorking with: {path_tmp_file}")
        file_downloader(url, path_tmp_file)

        # Retry while checksum is different.
        while md5(path_tmp_file) != file["checksum"].replace("md5:", ""):
            print(f"[WARNING] Checksum error when downloading {path_tmp_file}. We retry.")
            path_tmp_file.unlink()
            file_downloader(url, path_tmp_file)

        # Extract file in directory.
        path_to_unzip_or_move = Path(session_path)
        if ".zip" not in file["key"]:
            print(f"Move {path_tmp_file} to {path_to_unzip_or_move}.")
            shutil.move(path_tmp_file, Path(path_to_unzip_or_move, file["key"]))
        else:
            if "DCIM" in file["key"]:
                path_to_unzip_or_move = Path(session_path, "DCIM")
            elif "PROCESSED_DATA" in file["key"]:
                folder_name = file["key"].replace(".zip", "").replace("PROCESSED_DATA_", "")
                path_to_unzip_or_move = Path(session_path, "PROCESSED_DATA", folder_name)
            else:
                path_to_unzip_or_move = Path(session_path, file["key"].replace(".zip", ""))
            
            print(f"Unzip {path_tmp_file} to {path_to_unzip_or_move}.")
            with zipfile.ZipFile(path_tmp_file, 'r') as zip_ref:
                zip_ref.extractall(path_to_unzip_or_move)
            

    # Delete zip file and folder
    print(f"\nRemove {path_zip_session} folder.")
    for file in path_zip_session.iterdir():
        file.unlink()
    path_zip_session.rmdir()


def get_version_from_session_name(session_name: str) -> dict:
    """ Retrieve last version about a session with a session_name. """

    query = f'q=metadata.identifiers.identifier:"urn:{session_name}" metadata.related_identifiers.identifier:"urn:{session_name}"'
    r = requests.get(f"{ZENODO_LINK_WITHOUT_TOKEN}?{query}")

    version_json = {}
    if r.status_code == 404:
        print(f"Cannot access to {session_name}. Error 404")
        return version_json
    
    # Try to acces version. If all is good we have just one version, but if we have more or less than one version, we have an error.
    # We can have more than one version in deposit but here multiple version is traduct by multiple deposit.
    try:
        list_version = r.json()["hits"]["hits"]
        if len(list_version) > 1:
            print("Retrieve more than one version, abort.")
        elif len(list_version) == 0:
            print(f"No version found for {session_name}.")
        else:
            version_json = list_version[0]
    except:
        print(f"Cannot get version for {session_name}.")
    
    return version_json


def file_downloader(url: str, output_file: Path, params: dict = {}) -> None:
    """ Download file at output_file path. """
    isDownload, max_try = False, 0
    while not isDownload:
        try:
            r = requests.get(f"{url}", stream=True, params=params)
            total = int(r.headers.get('content-length', 0))

            with open(output_file, 'wb') as file, tqdm(total=total, unit='B', unit_scale=True) as bar:
                for data in r.iter_content(chunk_size=1000):
                    size = file.write(data)
                    bar.update(size)
            
            isDownload = True
        except KeyboardInterrupt:
            raise NameError("Stop iteration")
        except:
            print(traceback.format_exc(), end="\n\n")
            max_try += 1
            if max_try >= MAX_RETRY_TO_UPLOAD_DOWNLOAD_FILE: raise NameError("Abort due to max try")
            time.sleep(0.5)


def md5(fname: Path) -> str:
    """ Return md5 checksum of a file. """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
