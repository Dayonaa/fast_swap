from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.model_zoo.inswapper import INSwapper
from typing import Optional
import os
import urllib.request
import globals
from utilities import Utilities
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm


class FaceUtil:
    face_analyzer: Optional[FaceAnalysis] = None
    swapper: Optional[INSwapper] = None
    _model_path = "models/inswapper_128.onnx"
    _initialized: bool = False

    def __init__(self):
        if not FaceUtil._initialized:
            if self.initialize():
                FaceUtil._initialized = True


    def initialize(self) -> bool:
        try:
            if not Utilities.path_exists(FaceUtil._model_path):
                print(f"🔽 Mengunduh model dari {globals.INSWAPPER_MODEL_URL}...")
                os.makedirs("models", exist_ok=True)
                urllib.request.urlretrieve(globals.INSWAPPER_MODEL_URL, FaceUtil._model_path)
                print("✅ Model berhasil diunduh.")
            else:
                print("✔️ Model sudah tersedia secara lokal.")

            analyzer = FaceAnalysis(
                name='buffalo_l',
                root=os.getcwd(),
                allowed_modules=["landmark_3d_68", "landmark_2d_106", "detection", "recognition","genderage"],
            )
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            FaceUtil.face_analyzer = analyzer
            FaceUtil.swapper = get_model(FaceUtil._model_path, download=False)
            return True
        except Exception as e:
            print(f"❌ Terjadi error saat inisialisasi: {e}")
            return False
        
    def detect_faces(self,image_path: str):
        image = cv2.imread(image_path)
        faces = FaceUtil.face_analyzer.get(img=image)
        return image,faces
    
    def match_histogram(self,source, reference):
        """
        Transfer warna dari reference (target face) ke source (swapped face)
        """
        source_yuv = cv2.cvtColor(source, cv2.COLOR_BGR2YUV)
        reference_yuv = cv2.cvtColor(reference, cv2.COLOR_BGR2YUV)

        for i in range(3):
            source_channel = source_yuv[..., i]
            ref_channel = reference_yuv[..., i]

            mean_src, std_src = source_channel.mean(), source_channel.std()
            mean_ref, std_ref = ref_channel.mean(), ref_channel.std()

            # Hindari pembagian nol
            std_src = std_src if std_src > 1e-6 else 1

            matched = (source_channel - mean_src) * (std_ref / std_src) + mean_ref
            source_yuv[..., i] = np.clip(matched, 0, 255)

        return cv2.cvtColor(source_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    

    def swap_faces(self,source_face, target_image, target_face):
        swapper = FaceUtil.swapper
        return swapper.get(target_image, target_face, source_face)


    def swap_faces_on_folder(src_face):
        os.makedirs(globals.SWAPPED_FRAME_DIR, exist_ok=True)

        if src_face is None:
            print("❌ Tidak ada wajah sumber terdeteksi.")
            return

        frame_files = sorted(os.listdir(globals.EXTRACTED_FRAME_DIR))
        pbar = tqdm(frame_files, desc="🌀 Memproses frame", unit="frame")

        for filename in pbar:
            if not filename.endswith(".jpg"):
                continue

            frame_path = os.path.join(globals.EXTRACTED_FRAME_DIR, filename)
            t_image, t_faces = FaceUtil.detect_faces(frame_path)

            output_image = t_image.copy()

            if len(t_faces) > 0:
                for target_face in t_faces:
                    swapped = FaceUtil.swap_faces(src_face, output_image, target_face)
                    # update output_image supaya hasil swap diaplikasikan ke gambar selanjutnya
                    output_image = swapped

                status = "✅ swapped all faces"
            else:
                status = "⚠️ no face"

            out_path = os.path.join(globals.SWAPPED_FRAME_DIR, filename)
            cv2.imwrite(out_path, output_image)

            pbar.set_postfix({"frame": filename, "status": status})



    def scan_folder_and_count_faces(folder_path: str):
        image_files = sorted(os.listdir(folder_path))
        print(f"Scanning folder: {folder_path}")

        for filename in tqdm(image_files, desc="Scanning images"):
            if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
                continue

            image_path = os.path.join(folder_path, filename)
            image, faces = FaceUtil.detect_faces(image_path)

            face_count = len(faces) if faces else 0
            print(f"{filename}: {face_count} face(s) detected")


    def extract_face(face, idx:int, image: np.ndarray, margin: int = 10, size: tuple = (320, 320)) -> np.ndarray:
            # Ambil koordinat bbox dan konversi ke int
            x1, y1, x2, y2 = face.bbox.astype(int)[:4]

            # Dapatkan ukuran gambar
            h, w = image.shape[:2]

            # Tambahkan margin dan clip agar tetap di dalam gambar
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            # Crop dan resize
            face_img = image[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, size)

            # Buat folder kalau belum ada
            os.makedirs(globals.EXTRACTED_FRAME_DIR, exist_ok=True)

            filename = f"extracted_{idx}.jpg" 
            save_path = os.path.join(globals.EXTRACTED_FRAME_DIR, filename)

            # Simpan file
            cv2.imwrite(save_path, face_img)

            # return face_img
    
    def extract_aligned_face(face, image: np.ndarray, size: tuple = (112, 112)) -> np.ndarray:
        # Landmark 5 point (x, y)
        landmark = face.landmark_2d_106

        # Gunakan 5 landmark utama: [left_eye, right_eye, nose, left_mouth, right_mouth]
        idxs = [96, 97, 54, 76, 82]  # Index berdasarkan landmark_2d_106
        src = np.array([landmark[i] for i in idxs], dtype=np.float32)

        # Target posisi 5-point standar (dari InsightFace)
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        # Resize ke ukuran target
        dst *= size[0] / 112.0

        # Hitung affine transform dan align wajah
        M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        aligned_face = cv2.warpAffine(image, M, size, borderValue=0.0)

        return aligned_face
    
    def extract_aligned_face_2d(face, image: np.ndarray, size=(112,112)) -> np.ndarray:
        # Gunakan landmark 5 titik standar
        src = face.landmark_2d_5.astype(np.float32)

        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        dst *= size[0] / 112.0

        M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        aligned_face = cv2.warpAffine(image, M, size, borderValue=0.0)

        return aligned_face
            

    def detect_gender(self,face):
        if face is None:
            return None
        return "Male" if face.sex == "M" else "Female"

    def crop_face(self,face, image):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        h, w = image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            # print("❌ Koordinat wajah di luar batas gambar atau tidak valid.")
            return None

        return image[y1:y2, x1:x2]

    def test_extract_face(self, img_path: str):
            try:
                image = cv2.imread(filename=img_path)
                faces = FaceUtil.face_analyzer.get(img=image,max_num=10)
                if not faces:
                    print("Tidak ada wajah terdeteksi.")
                    return

                for i, face in enumerate(faces):
                    print(f'Gender {FaceUtil.detect_gender(face)}')
                    FaceUtil.extract_face(face,i,image, size=(320,320))
                    # cv2.imwrite(f"face_{i+1}.jpg", face_img)
            except Exception as e:
                print(e)

    def test_swap(self, src_path):
        try:
            img, faces = FaceUtil.detect_faces(src_path)
            t_img, targets = FaceUtil.detect_faces("assets/hh.jpg")

            if len(faces) == 0 or len(targets) == 0:
                print("Tidak ada wajah terdeteksi.")
                return

            source_face = faces[0]
            target_face = targets[0]
            output_img = t_img.copy() 
            for i, t_face in enumerate(targets):
                output_img = FaceUtil.swap_faces(source_face, output_img, t_face)
            
            crop_target=FaceUtil.crop_face(target_face,t_img)
            fix_color=FaceUtil.match_histogram(source=output_img,reference=crop_target)
            cv2.imwrite("final.jpg", fix_color)
        except Exception as e:
            print(f"Error: {e}")

      
