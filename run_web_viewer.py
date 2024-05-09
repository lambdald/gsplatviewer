import tyro
from pathlib import Path
import os
import shutil
from gaussian_model import GaussianModel

BASE_DIR = Path(__file__).parent
VIEW_MODEL_PATH = BASE_DIR / "models/model.splat"
# DEFAULT_MODEL_PATH = BASE_DIR / "models/default.splat"

def main(model_path: Path, host: str = "0.0.0.0", port: int = 9999):
    # if model_path is None:
    #     if DEFAULT_MODEL_PATH.exists():
    #         model_path = DEFAULT_MODEL_PATH
    #         print(f"use default model {DEFAULT_MODEL_PATH}")
    #     else:
    #         raise ValueError('please input a 3dgs model path')
    

    VIEW_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

    if model_path.suffix == ".ply":
        print(f"Convert 3dgs point cloud {model_path} to splat file {VIEW_MODEL_PATH}")
        data = GaussianModel.from_file(model_path)
        data.to_splat_file(VIEW_MODEL_PATH)
    else:
        print(f"Copy 3dgs splat file {model_path} to {VIEW_MODEL_PATH}...")
        shutil.copy(model_path, VIEW_MODEL_PATH)


    size = VIEW_MODEL_PATH.stat().st_size / 1024 / 1024
    print(f'model size={size:.2f} MB')

    os.chdir(BASE_DIR)
    try:
        os.system(f"python -m http.server {port} --bind {host}")
    except Exception as e:
        VIEW_MODEL_PATH.unlink(VIEW_MODEL_PATH)


if __name__ == "__main__":
    tyro.cli(main)
