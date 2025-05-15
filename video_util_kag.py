import os
import cv2
import globals
from face_util_kag import FaceUtil
from tqdm import tqdm
import shutil
from multiprocessing import Process
import copy
from google.colab.patches import cv2_imshow
import numpy as np

class VideoUtil:
    def test():
        print("")

    import shutil  # tambahkan ini di bagian atas file jika belum ada

    @staticmethod
    def extract_video_frames(video_path, duration_sec=None):
        # Hapus folder extracted frames jika ada
        if os.path.exists(globals.EXTRACTED_FRAME_DIR):
            shutil.rmtree(globals.EXTRACTED_FRAME_DIR)
        os.makedirs(globals.EXTRACTED_FRAME_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Gagal membuka video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Ambil total frame video
        video_duration_sec = total_frames / fps  # Hitung durasi video dalam detik

        # Jika duration_sec None, gunakan durasi video
        if duration_sec is None:
            duration_sec = video_duration_sec

        # Hitung total frame berdasarkan durasi yang diinginkan
        total_frames_to_extract = int(duration_sec * fps)

        print(f"⚙️ Durasi video: {video_duration_sec:.2f} detik. Menyimpan {total_frames_to_extract} frame.")

        count = 0
        while count < total_frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(globals.EXTRACTED_FRAME_DIR, f"frame_{count:05d}.jpg"), frame)
            count += 1

        cap.release()
        print(f"✅ {count} frame (durasi {duration_sec} detik) disimpan ke '{globals.EXTRACTED_FRAME_DIR}'.")


    @staticmethod
    def check_gender():
        face_util = FaceUtil()
        image_files = sorted(os.listdir(globals.EXTRACTED_FRAME_DIR))

        for img_name in image_files:
            img_path = os.path.join(globals.EXTRACTED_FRAME_DIR, img_name)

            image, faces = face_util.detect_faces(img_path)  # <- return tuple: image, faces

            print(f"{img_name} -> {len(faces)} face(s)")

            for i, face in enumerate(faces):
                gender = face_util.detect_gender(face)
                print(gender)


    @staticmethod
    def swap_video(src_image_path):
        face_util = FaceUtil()
        image_files = sorted(os.listdir(globals.EXTRACTED_FRAME_DIR))

        # Ambil source face
        src_img, src_faces = face_util.detect_faces(src_image_path)
        if len(src_faces) == 0:
            print("❌ Tidak ada wajah sumber.")
            return
        source_face = src_faces[0]

        # Ambil target referensi untuk histogram dari frame pertama yang punya wajah
        reference_crop = None
        for img_name in image_files:
            img_path = os.path.join(globals.EXTRACTED_FRAME_DIR, img_name)
            img, faces = face_util.detect_faces(img_path)
            if faces:
                reference_crop = face_util.crop_face(faces[0], img)
                break

        # Hapus folder swapped frames jika ada
        if os.path.exists(globals.SWAPPED_FRAME_DIR):
            shutil.rmtree(globals.SWAPPED_FRAME_DIR)
        os.makedirs(globals.SWAPPED_FRAME_DIR, exist_ok=True)

        pbar = tqdm(enumerate(image_files), total=len(image_files), desc="🎞️ Proses swapping", unit="frame")

        preview_images = []

        for idx, img_name in pbar:
            img_path = os.path.join(globals.EXTRACTED_FRAME_DIR, img_name)
            image, faces = face_util.detect_faces(img_path)
            output_img = image.copy()

            if faces:
                for face in faces:
                    gender = face_util.detect_gender(face)
                    if gender == "Female":
                        swapped = face_util.swap_faces(source_face, output_img, face)
                        swapped = face_util.match_histogram(swapped, reference_crop)
                        output_img = swapped

            out_path = os.path.join(globals.SWAPPED_FRAME_DIR, img_name)
            cv2.imwrite(out_path, output_img)

            # 🖼️ Simpan ke daftar untuk grid preview
            if idx % 100 == 0:
                thumb = cv2.resize(output_img, (320, 180))  # kecilkan untuk grid
                preview_images.append(thumb)

            # 📦 Tampilkan grid setiap 6 preview
            if len(preview_images) == 6:
                print(f"\n📸 Grid preview (frame ke-{idx})")
                grid_row1 = np.hstack(preview_images[:3])
                grid_row2 = np.hstack(preview_images[3:])
                grid = np.vstack([grid_row1, grid_row2])
                cv2_imshow(grid)
                preview_images = []

        # Hapus folder extracted frame setelah selesai
        shutil.rmtree(globals.EXTRACTED_FRAME_DIR)
        print(f"\n🧹 Folder {globals.EXTRACTED_FRAME_DIR} telah dihapus.")



    @staticmethod
    def create_video_from_frames(fps=30):
        frame_files = sorted(os.listdir(globals.SWAPPED_FRAME_DIR))
        if len(frame_files) == 0:
            print("❌ Tidak ada frame ditemukan di folder.")
            return

        first_frame_path = os.path.join(globals.SWAPPED_FRAME_DIR, frame_files[0])
        first_frame = cv2.imread(first_frame_path)

        if first_frame is None:
            print("❌ Gagal membaca frame pertama.")
            return

        height, width, _ = first_frame.shape

        # Pastikan folder output ada
        output_dir = os.path.dirname(globals.OUTPUT_VIDEO)
        os.makedirs(output_dir, exist_ok=True)

        # Inisialisasi VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(globals.OUTPUT_VIDEO, fourcc, fps, (width, height))

        for filename in frame_files:
            if filename.lower().endswith((".jpg", ".png")):
                frame_path = os.path.join(globals.SWAPPED_FRAME_DIR, filename)
                frame = cv2.imread(frame_path)

                if frame is not None:
                    out.write(frame)
                else:
                    print(f"⚠️ Gagal membaca frame {filename}, dilewati.")

        out.release()
        print(f"✅ Video selesai dibuat: {globals.OUTPUT_VIDEO}")



    @staticmethod
    def swap_video_parallel(src_image_path):
        # src_img, src_faces = FaceUtil.detect_faces(src_image_path)
        # if len(src_faces) == 0:
        #     print("❌ Tidak ada wajah sumber.")
        #     return
        # source_face = src_faces[0]

        image_files = sorted(os.listdir(globals.EXTRACTED_FRAME_DIR))
        os.makedirs(globals.SWAPPED_FRAME_DIR, exist_ok=True)

        mid = len(image_files) // 2
        list_0 = image_files[:mid]
        list_1 = image_files[mid:]

        # src_face_data = source_face.__dict__

        # ✅ Panggil langsung FaceUtil.swap_worker karena sudah staticmethod
        p0 = Process(target=FaceUtil.swap_worker, args=(0, list_0, copy.deepcopy(src_image_path)))
        p1 = Process(target=FaceUtil.swap_worker, args=(1, list_1, copy.deepcopy(src_image_path)))
        p0.start()
        p1.start()
        p0.join()
        p1.join()


    @staticmethod
    def swap_video_sequential_dual_gpu(src_image_path):
        face_util = FaceUtil()
        src_img, src_faces = face_util.detect_faces(src_image_path)
        if len(src_faces) == 0:
            print("❌ Tidak ada wajah sumber.")
            return

        image_files = sorted(os.listdir(globals.EXTRACTED_FRAME_DIR))
        os.makedirs(globals.SWAPPED_FRAME_DIR, exist_ok=True)

        mid = len(image_files) // 2
        list_0 = image_files[:mid]
        list_1 = image_files[mid:]

        # Jalankan swap_worker di GPU 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        face_util.swap_worker(0, list_0, src_image_path)

        # Jalankan swap_worker di GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        face_util.swap_worker(1, list_1, src_image_path)
