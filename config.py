from pathlib import Path
import sys
import os

if getattr(sys, 'frozen', False):
    # 如果是通过 PyInstaller 打包后的可执行文件运行，则获取临时文件夹路径
    current_dir = Path(sys._MEIPASS)
else:
    # 否则获取当前脚本所在目录路径
    current_dir = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = current_dir.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
SEGMENT_MODEL_DIR = ROOT / 'weights' / 'segment'
POSE_MODEL_DIR = ROOT / 'weights' / 'pose'
OUTPUTS_MODEL_DIR = ROOT / 'outputs'

MODEL_LIST = {
    "yolov8n": DETECTION_MODEL_DIR / "yolov8n.pt",
    "yolov8s": DETECTION_MODEL_DIR / "yolov8s.pt",
    "yolov8m": DETECTION_MODEL_DIR / "yolov8m.pt",
    "yolov8l": DETECTION_MODEL_DIR / "yolov8l.pt",
    "yolov8x": DETECTION_MODEL_DIR / "yolov8x.pt",
    "yolov5s": DETECTION_MODEL_DIR / "yolov5s.pt",
    "yolov5m": DETECTION_MODEL_DIR / "yolov5m.pt",
    "yolov5l": DETECTION_MODEL_DIR / "yolov5l.pt",
    "yolov5x": DETECTION_MODEL_DIR / "yolov5x.pt",
    "mmdetection": DETECTION_MODEL_DIR / "mmdetection.pt",
}

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    'yolov5n.pt',
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt",
    'yolov5n6.pt',
    'yolov5s6.pt',
    'yolov5m6.pt',
    'yolov5l6.pt',
    'yolov5x6.pt',
    "mmdetection.pt",
    ]

SEGMENT_MODEL_LIST = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
    "mmdetection-seg.pt",]

POSE_MODEL_LIST = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
    "mmdetection-pose.pt",]

# %%
def get_relative_paths(OUTPUTS_MODEL_DIR):
    pt_files = []
    for root, dirs, files in os.walk(OUTPUTS_MODEL_DIR):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.relpath(os.path.join(root, file), OUTPUTS_MODEL_DIR))
    return pt_files


#%%

OUTPUTS_MODEL_LIST = get_relative_paths(OUTPUTS_MODEL_DIR)