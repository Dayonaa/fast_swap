from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper


ONNXX_PATH = "models/inswapper_128.onnx"
VIDEO_PATH = "assets/haha.mp4"
TARGET_MULTI_FACE_PATH = [
    "assets/family.jpg",
]
DUMMY_SRC = [
    "assets/lon.jpg",
    "assets/hh.jpg",
]
ELSA_PATH = ["assets/elsa.jpeg", "assets/elsa2.jpg"]
NN_PATH = ["assets/n1.jpg", "assets/n2.png"]
EXTRACTED_FRAME_DIR = "EXTRACTED_FRAME"
SWAPPED_FRAME_DIR = "SWAPPED_FRAME"
VIDEO_OUTPUT = "OUTPUT/final.mp4"


SWAPPER_APP: INSwapper
ANTELOP_V2_APP: FaceAnalysis
BUFFALO_APP: FaceAnalysis
