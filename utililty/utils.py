import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
import shutil
from tqdm.notebook import tqdm
import globals


class Utils:
    def __init__(self):
        self.onnx_path = globals.ONNXX_PATH
        self.extracted_frame_dir = globals.EXTRACTED_FRAME_DIR
        self.swapped_frame_dir = globals.SWAPPED_FRAME_DIR
        self.video_output = globals.VIDEO_OUTPUT

    def load_models(self):
        print("\U0001f527 Loading models (CPU)...")
        globals.SWAPPER_APP = get_model(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        globals.ANTELOP_V2_APP = FaceAnalysis(
            name="antelopev2",
            providers=["CPUExecutionProvider"],
            root=os.getcwd(),
            allowed_modules=[
                "landmark_3d_68",
                "landmark_2d_106",
                "detection",
                "recognition",
                "genderage",
            ],
        )
        globals.BUFFALO_APP = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=os.getcwd(),
            allowed_modules=[
                "landmark_3d_68",
                "landmark_2d_106",
                "detection",
                "recognition",
                "genderage",
            ],
        )
        globals.ANTELOP_V2_APP.prepare(ctx_id=0, det_size=(640, 640))
        globals.BUFFALO_APP.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ Models loaded.")

    @staticmethod
    def show_image(img: str | np.ndarray, figsize=(4, 4)) -> None:
        if isinstance(img, str):
            if not os.path.exists(img):
                print(f"❌ File tidak ditemukan: {img}")
                return
            img = cv2.imread(img)
            if img is None:
                print(f"❌ Gagal membaca gambar dari path: {img}")
                return
        elif not isinstance(img, np.ndarray):
            print("❌ Input harus berupa path (str) atau gambar (np.ndarray).")
            return

        # Konversi ke RGB kalau gambar 3-channel (warna)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=figsize)
        if img.ndim == 2:  # Grayscale image
            plt.imshow(img, cmap="gray")
        else:  # RGB image
            plt.imshow(img)
        plt.axis("off")
        plt.show()

    @staticmethod
    def load_image(img: Union[str, np.ndarray]) -> np.ndarray | None:
        if isinstance(img, str):
            if not os.path.exists(img):
                print(f"❌ File tidak ditemukan: {img}")
                return None
            img = cv2.imread(img)
            if img is None:
                print(f"❌ Gagal membaca gambar dari path: {img}")
            return img
        elif isinstance(img, np.ndarray):
            return img
        else:
            print("❌ Input harus path (str) atau np.ndarray.")
            return None

    @staticmethod
    def load_image_new(img: Union[str, np.ndarray]) -> np.ndarray | None:
        if isinstance(img, str):
            if not os.path.exists(img):
                print(f"❌ File tidak ditemukan: {img}")
                return None
            img = cv2.imread(img)
            if img is None:
                print(f"❌ Gagal membaca gambar dari path: {img}")
                return None
            # ✅ Convert BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(img, np.ndarray):
            # Jika array grayscale atau bukan RGB, ubah
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return img
        else:
            print("❌ Input harus path (str) atau np.ndarray.")
            return None

    @staticmethod
    def extract_video_frames(self, video_path, out_dir, duration_sec=None):
        if out_dir is None:
            out_dir = self.extracted_frame_dir

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Gagal membuka video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_sec = total_frames / fps

        if duration_sec is None:
            duration_sec = video_duration_sec

        total_frames_to_extract = int(min(duration_sec * fps, total_frames))
        print(
            f"⚙️ Durasi video: {video_duration_sec:.2f} detik. Menyimpan {total_frames_to_extract} frame."
        )

        frame_count = 0
        pbar = tqdm(total=total_frames_to_extract, desc="🔍 Ekstraksi frame")

        while frame_count < total_frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(out_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        print(
            f"✅ {frame_count} frame (durasi {duration_sec:.2f} detik) disimpan ke '{out_dir}'."
        )

    @staticmethod
    def create_video_from_frames(self, frames_path, out_dir, fps=30):
        if out_dir is None:
            out_dir = self.video_output

        if frame_path is None:
            out_dir = self.swapped_frame_dir

        frame_files = sorted(os.listdir(frames_path))

        if len(frame_files) == 0:
            print("❌ Tidak ada frame ditemukan di folder.")
            return

        first_frame_path = os.path.join(frames_path, frame_files[0])
        first_frame = cv2.imread(first_frame_path)

        if first_frame is None:
            print("❌ Gagal membaca frame pertama.")
            return

        height, width, _ = first_frame.shape

        output_dir = os.path.dirname(out_dir)
        os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_dir, fourcc, fps, (width, height))

        for filename in frame_files:
            if filename.lower().endswith((".jpg", ".png")):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
                else:
                    print(f"⚠️ Gagal membaca frame {filename}, dilewati.")

        out.release()
        print(f"✅ Video selesai dibuat: {out_dir}")
