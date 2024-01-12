# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform
import subprocess
import time
from pathlib import Path

import requests
import torch

from pprint import pprint  # DEB


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(file, repo='WongKinYiu/yolov7'):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", '').lower())

    if not file.exists():
        try:
            # Get the latest release tag from the GitHub API, and
            # download the latest version of the Yolov7 model.
            api_url = f"https://api.github.com/repos/{repo}/releases"
            response = requests.get(api_url)
            # Raise an exception for 4xx and 5xx status codes.
            response.raise_for_status()
            releases = response.json()

            # Extract tags from the GitHub API response
            tags = [release["tag_name"] for release in releases]
            tag = tags[0]  # Required tag is v1.0.
            msg_err = f"v1.0 tag not found in {tags}."
            assert tag == "v1.0", msg_err
        except Exception as e:  # Fallback plan.
            print(f"Error fetching releases: {e}")

        # List of available assets for the
        # yolov7 model in the v1.0 release.
        assets = [
            "yolov7.pt", "yolov7-tiny.pt", 
            "yolov7x.pt", "yolov7-d6.pt", 
            "yolov7-e6.pt", "yolov7-e6e.pt", 
            "yolov7-w6.pt"
        ]

        # Extract the name of the required asset 
        # from the file path.
        asset_name = file.name
        if asset_name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/download/tag/{tag}"
            redundant = False
            try:
                # Download the Yolov7 model from the
                # GitHub release page.
                git_url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
                print(f"Downloading {git_url} to {file}...")
                torch.hub.download_url_to_file(
                    git_url, 
                    file, 
                    progress=True
                )
                print(f"File successfully downloaded to: {file}")  # DEB
                # Assertion check to ensure that the file
                # was downloaded successfully.
                assert file.exists() and file.stat().st_size > 1E6
            except Exception as e:
                # If the file download fails, try downloading
                # the file from the Google Cloud Platform (GCP) Storage 
                # bucket.
                print(f"Download error: {e}")
                assert redundant, "No secondary mirror."
                gcp_url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                print(f"Downloading {gcp_url} to {file}...")
                os.system(f"curl -L {url} -o {file}")  # torch.hub.download_url_to_file(url, weights)
            finally:
                if not file.exists() or file.stat().st_size < 1E6:
                    # Remove partial downloads if download fails.
                    file.unlink(missing_ok=True)
                    print(f'ERROR: Download failure: {msg}')
                print('')
    
    return


def gdrive_download(id='', file='tmp.zip'):
    # Downloads a file from Google Drive. from yolov7.utils.google_utils import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')  # gdrive cookie
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        os.system(f'unzip -q {file}')  # unzip
        file.unlink()  # remove zip to free space

    print(f'Done ({time.time() - t:.1f}s)')
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
