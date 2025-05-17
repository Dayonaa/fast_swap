import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from insightface.app import FaceAnalysis
from insightface.app.face_analysis import Face
from insightface.model_zoo import get_model
import warnings
import onnxruntime as ort
from globals import SWAPPED_FRAME_DIR
from utililty.utils import Utils
from tqdm.notebook import tqdm
from insightface.model_zoo.inswapper import INSwapper

warnings.simplefilter(action="ignore", category=FutureWarning)
ort.set_default_logger_severity(3)


class FaceUtils:
    def __init__(self, app: FaceAnalysis, swapper: INSwapper):
        self.app = app
        self.swapper = swapper

    def crop_face(
        self, img_path: np.ndarray | str, margin=0.2, show=False
    ) -> Tuple[np.ndarray, Face] | None:
        img = Utils.load_image(img_path)
        if img is None:
            print(f"❌ Gagal membaca gambar: {img_path}")
            return None, None

        faces = self.app.get(img)
        if not faces:
            print("⚠️ Tidak ada wajah terdeteksi.")
            return None, None

        face = faces[0]
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)

        dx = int((x2 - x1) * margin)
        dy = int((y2 - y1) * margin)
        x1_m = max(0, x1 - dx)
        y1_m = max(0, y1 - dy)
        x2_m = min(w, x2 + dx)
        y2_m = min(h, y2 + dy)

        face_crop = img[y1_m:y2_m, x1_m:x2_m]

        if show:
            plt.figure(figsize=(4, 4))
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            plt.imshow(face_rgb)
            plt.title("Cropped Face")
            plt.axis("off")
            plt.show()
            return None

        return face_crop, face

    def get_face(self, img_path: str, show: bool = False) -> List[Face] | None:
        """
        Deteksi semua wajah dari gambar dan kembalikan daftar Face object.

        Args:
            img_path (str): Path ke file gambar.
            show (bool): Jika True, tampilkan gambar dengan kotak deteksi.

        Returns:
            List[Face] | None: List wajah yang terdeteksi atau None jika gagal.
        """
        img = Utils.load_image(img_path)
        if img is None:
            print(f"❌ Gagal membaca gambar: {img_path}")
            return None

        img_draw = img.copy()
        faces: List[Face] = self.app.get(img)

        if not faces:
            print("⚠️ Tidak ada wajah terdeteksi.")
            return None

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show:
            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()

        return faces

    def create_face_mask(
        self,
        src: np.ndarray | str,
        inner_face_only: bool = False,
        erode: int = 0,
        feather: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Membuat mask wajah berdasarkan landmark InsightFace.
        Otomatis pakai landmark 3D jika tersedia.

        Args:
            src (np.ndarray | str): Gambar sumber
            inner_face_only (bool): True = hanya bagian wajah dalam (tanpa dahi dan rahang)
            erode (int): Jumlah piksel erosi untuk mengecilkan area mask
            feather (int): Radius feathering (Gaussian blur)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (mask, masked image)
        """
        img = Utils.load_image(src)
        if img is None:
            return None

        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        face = self.get_face(img_path=src)
        if not face or face[0] is None:
            return None

        face = face[0]

        # Pilih landmark terbaik yang tersedia
        if hasattr(face, "landmark_3d_68"):
            points = face.landmark_3d_68[:, :2].astype(np.int32)
            if inner_face_only:
                inner_idxs = np.array(
                    list(range(17, 27))  # alis & hidung atas
                    + list(range(27, 36))  # hidung
                    + list(range(36, 48))  # mata
                    + list(range(48, 68))  # mulut
                )
                points = points[inner_idxs]
        elif hasattr(face, "landmark_2d_106"):
            points = face.landmark_2d_106.astype(np.int32)
            if inner_face_only and points.shape[0] >= 68:
                points = points[17:68]
        else:
            points = face.kps.astype(np.int32)

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        if erode > 0:
            kernel = np.ones((erode, erode), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        if feather > 0:
            ksize = feather if feather % 2 == 1 else feather + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        overlay = img.copy()
        green = np.array([0, 255, 0], dtype=np.uint8)

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0,
                (0.3 * green[c] + 0.7 * img[:, :, c]).astype(np.uint8),
                img[:, :, c],
            )

        return mask, overlay

    def create_face_mask_new(
        self,
        src: np.ndarray | str,
        inner_face_only: bool = False,
        erode: int = 0,
        feather: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Membuat mask wajah berdasarkan landmark InsightFace.

        Args:
            src (np.ndarray | str): Gambar sumber (np.ndarray atau path)
            inner_face_only (bool): True = hanya bagian wajah dalam (mata, hidung, mulut)
            erode (int): Jumlah piksel erosi
            feather (int): Radius feathering (blur mask)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (mask, overlay) atau None jika gagal
        """
        img = Utils.load_image(src)
        if img is None:
            return None

        face_list = self.app.get(img)
        if not face_list:
            return None

        face = face_list[0]

        if hasattr(face, "landmark_3d_68"):
            points = face.landmark_3d_68[:, :2].astype(np.int32)
            if inner_face_only:
                inner_idxs = np.array(
                    list(range(17, 27))  # alis & hidung atas
                    + list(range(27, 36))  # hidung
                    + list(range(36, 48))  # mata
                    + list(range(48, 68))  # mulut
                )
                points = points[inner_idxs]
        elif hasattr(face, "landmark_2d_106"):
            points = face.landmark_2d_106.astype(np.int32)
            if inner_face_only and points.shape[0] >= 68:
                points = points[17:68]
        elif hasattr(face, "kps"):
            points = face.kps.astype(np.int32)
        else:
            return None

        # Buat mask dari convex hull
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        if erode > 0:
            kernel = np.ones((erode, erode), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        if feather > 0:
            ksize = feather if feather % 2 == 1 else feather + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Buat overlay hijau semi transparan untuk debug visual
        overlay = img.copy()
        green = np.array([0, 255, 0], dtype=np.uint8)

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0,
                (0.3 * green[c] + 0.7 * img[:, :, c]).astype(np.uint8),
                img[:, :, c],
            )

        return mask, overlay

    def swap_face(
        self,
        src: np.ndarray | str,
        target: np.ndarray | str,
        out_dir: str = SWAPPED_FRAME_DIR,
        show_progress: bool = False,
        gender: str = "F",  # 'F' atau 'M'
        min_score: float = 0.80,
        file_name: str = "swapped",
        use_mask: bool = False,
        inner_face_only: bool = True,
        erode: int = 5,
        feather: int = 15,
        save: bool = True,
    ) -> np.ndarray | None:
        """
        Ganti wajah pada gambar `src` dengan wajah dari `target`, lalu simpan hasil ke `out_dir`.

        Args:
            src (np.ndarray | str): Gambar sumber yang wajahnya akan DIGANTI
            target (np.ndarray | str): Gambar wajah yang akan DITANAMKAN
            out_dir (str): Folder output hasil swap
            show_progress (bool): Jika True, tampilkan progress bar tqdm
            gender (str): Filter gender wajah di `src` ('F' atau 'M')
            min_score (float): Minimum confidence score (0.0 - 1.0)
        """

        if src is None or target is None:
            print("❌ Gagal memuat gambar.")
            return

        faces_src = self.get_face(src)
        faces_target = self.get_face(target)

        if not faces_src or not faces_target:
            print("⚠️ Tidak ditemukan wajah pada salah satu gambar.")
            return

        # Filter wajah di src berdasarkan gender dan score
        filtered_faces_src = [
            face
            for face in faces_src
            if getattr(face, "sex", None) == gender
            and getattr(face, "det_score", 0.0) >= min_score
        ]

        if not filtered_faces_src:
            print(
                f"⚠️ Tidak ada wajah di `src` yang cocok dengan filter gender='{gender}' dan score>={min_score}."
            )
            if save:
                # Tetap simpan gambar asli sebagai fallback
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{file_name}.jpg")
                cv2.imwrite(out_path, src)
                print(
                    f"📷 Gambar asli disimpan karena tidak ada wajah yang cocok: {out_path}"
                )
                return
            return
        target_face = faces_target[0]
        swapped_img = Utils.load_image(src).copy()

        iterator = (
            tqdm(filtered_faces_src, desc="🔁 Swapping faces")
            if show_progress
            else filtered_faces_src
        )

        for source_face in iterator:
            swapped_temp = self.swapper.get(
                swapped_img, source_face, target_face, paste_back=True
            )

            if use_mask:
                mask_result = self.create_face_mask_new(
                    src=swapped_img,
                    inner_face_only=inner_face_only,
                    erode=erode,
                    feather=feather,
                )

                if mask_result is None:
                    print("❌ Mask gagal dibuat")
                    continue

                mask, _ = mask_result

                # 🧠 Gabungkan hanya area mask dari wajah yang sudah diswap
                mask_3ch = cv2.merge([mask] * 3)  # convert mask ke 3 channel
                swapped_img = np.where(mask_3ch > 0, swapped_temp, swapped_img)
            else:
                swapped_img = swapped_temp
        if save:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{file_name}.jpg")
            cv2.imwrite(out_path, swapped_img)
            print(f"✅ Swap selesai. Hasil disimpan di: {out_path}")
            return
        return swapped_img
