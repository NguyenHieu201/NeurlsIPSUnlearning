import os

import gdown
import tarfile


class Downloader:
    data_path = "./data/UTTKFace.tar"
    img_path = "./data/"
    data_id = "11epWGN3_1Z4yZZQRVCJovXDr6BtDTlmK"

    def __init__(self) -> None:
        pass

    def download_data():
        if not os.path.exists(Downloader.data_path):
            os.makedirs("./data")
        gdown.download(id=Downloader.data_id, output=Downloader.data_path)
        with tarfile.open(Downloader.data_path, "r") as tar_file:
            tar_file.extractall(Downloader.img_path)


if __name__ == "__main__":
    Downloader.download_data()
