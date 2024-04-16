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

YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"
YOLOvcs2 = DETECTION_MODEL_DIR / "sbest.pt"
YOLOvcs1 = DETECTION_MODEL_DIR / "nlast.pt"
YOLOvcs3 = DETECTION_MODEL_DIR / "last.pt"

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "nlast.pt",
    "sbest.pt",
    "last.pt"]

SEGMENT_MODEL_LIST = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt"]

POSE_MODEL_LIST = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt"]
