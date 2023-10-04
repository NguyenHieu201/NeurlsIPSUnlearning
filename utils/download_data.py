import gdown
import tarfile


class Downloader:
    data_path = "./data/UTTKFace.tar"
    img_path = "./data/UTTKFace"
    data_id = "11epWGN3_1Z4yZZQRVCJovXDr6BtDTlmK"

    def __init__(self) -> None:
        pass

    def download_data():
        gdown.download(Downloader.data_id, Downloader.data_path)
        with tarfile.open(Downloader.data_path, "r") as tar_file:
            tar_file.extractall(Downloader.img_path)


if __name__ == "__main__":
    Downloader.download_data()
