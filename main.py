# ====== IMPORTS ======
from face_util.face_utils import FaceUtils
from utililty.utils import Utils
import globals as G
import cv2

utils = Utils()


utils.load_models()
BUFFALO_APP = FaceUtils(G.BUFFALO_APP)
ANTELOPE_APP = FaceUtils(G.ANTELOP_V2_APP)


# img = cv2.imread(G.ELSA_PATH[0])
# faces = BUFFALO_APP.app.get(img)[0]
# print(faces)
# face = BUFFALO_APP.get_faces(
#     src_img=G.ELSA_PATH[0], aligned_save_dir="align", image_id_prefix="elsa"
# )
# print(face)
