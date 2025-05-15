import os
import urllib.request
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.model_zoo.inswapper import INSwapper
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
import shutil
from multiprocessing import Process, Manager
import time

model_path = "models/inswapper_128.onnx"
INSWAPPER_MODEL_URL = "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"

# face_analyzer is (FaceAnalysis, INSwapper)
face_analyzer: Optional[Tuple[FaceAnalysis, INSwapper]] = None



def initialize() -> bool:
    try:
        if not os.path.exists(model_path):
            print(f"🔽 Mengunduh model dari {INSWAPPER_MODEL_URL}...")
            urllib.request.urlretrieve(INSWAPPER_MODEL_URL, model_path)
            print("✅ Model berhasil diunduh.")
        else:
            print("✔️ Model sudah tersedia secara lokal.")

        init_analyzer()
        return face_analyzer is not None
    except Exception as e:
        print(f"❌ Terjadi error saat inisialisasi: {e}")
        return False

def init_analyzer():
    try:
        print("📦 Inisialisasi model...")
        global face_analyzer
        analyzer =FaceAnalysis(
        name='buffalo_l',
        allowed_modules=["landmark_3d_68", "landmark_2d_106","detection","recognition"],
        )
        analyzer.prepare(ctx_id=0,det_size=(640, 640))  # -1 = CPU
        swapper = get_model(model_path, download=False)
        print("✅ Model siap digunakan.")
        face_analyzer = (analyzer, swapper)
    except Exception as e:
        print(f"❌ Terjadi error saat inisialisasi: {e}")


def detect_faces(image_path: str):
    analyzer, _ = face_analyzer
    image = cv2.imread(image_path)
    faces = analyzer.get(image)
    return image, faces

def extract_face(face, image):
    bbox = face.bbox.astype(int)
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def swap_faces(source_face, target_image, target_face):
    _, swapper = face_analyzer
    return swapper.get(target_image, target_face, source_face)


def match_histogram(source, reference):
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


def extract_video_frames(video_path, output_dir="vid_frame", duration_sec=None):
    os.makedirs(output_dir, exist_ok=True)
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
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.jpg"), frame)
        count += 1

    cap.release()
    print(f"✅ {count} frame (durasi {duration_sec} detik) disimpan ke '{output_dir}'.")


def swap_faces_on_frames(frames_dir, base_output="swapped_frames"):
    os.makedirs(base_output, exist_ok=True)

    # Deteksi wajah sumber
    s_img, s_face = detect_faces("/content/h2.png")
    if len(s_face) == 0:
        print("❌ Tidak ada wajah sumber terdeteksi.")
        return

    source_face = s_face[0]
    src_crop = extract_face(source_face, s_img)

    first_target_crop = None

    frame_files = sorted(os.listdir(frames_dir))
    pbar = tqdm(frame_files, desc="🌀 Memproses frame", unit="frame")

    for filename in pbar:
        if not filename.endswith(".jpg"):
            continue

        frame_path = os.path.join(frames_dir, filename)
        t_image, t_faces = detect_faces(frame_path)

        output_image = t_image.copy()

        if len(t_faces) > 0:
            target_face = t_faces[0]
            if first_target_crop is None:
                first_target_crop = extract_face(target_face, t_image)

            swapped = swap_faces(source_face, output_image, target_face)

            if first_target_crop is not None:
                swapped = match_histogram(swapped, first_target_crop)

            output_image = swapped
            status = "✅ swapped"
        else:
            status = "⚠️ no face"

        out_path = os.path.join(base_output, filename)
        cv2.imwrite(out_path, output_image)

        # Update status tanpa menambah baris baru
        pbar.set_postfix({"frame": filename, "status": status})


def create_video_from_frames(frames_dir, output_video_path, fps=30):
    # Ambil file gambar pertama untuk menentukan resolusi video
    frame_files = sorted(os.listdir(frames_dir))
    if len(frame_files) == 0:
        print("❌ Tidak ada frame ditemukan di folder.")
        return
    
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        print("❌ Gagal membaca frame pertama.")
        return

    # Ambil resolusi dari frame pertama
    height, width, _ = first_frame.shape

    # Inisialisasi VideoWriter untuk menyimpan video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # untuk format mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop untuk menulis setiap frame ke dalam video
    for filename in frame_files:
        if filename.endswith(".jpg"):  # Pastikan hanya file gambar yang diproses
            frame_path = os.path.join(frames_dir, filename)
            frame = cv2.imread(frame_path)

            if frame is not None:
                out.write(frame)  # Menulis frame ke video
            else:
                print(f"⚠️ Gagal membaca frame {filename}, dilewati.")

    out.release()  # Menyelesaikan dan menyimpan video
    print(f"✅ Video selesai dibuat: {output_video_path}")


def swap_faces_on_frames_partial(frame_files, source_face, first_target_crop, frames_dir, output_dir, progress_dict):
    os.makedirs(output_dir, exist_ok=True)
    total_frames = len(frame_files)

    for idx, filename in enumerate(frame_files):
        if not filename.endswith(".jpg"):
            continue

        frame_path = os.path.join(frames_dir, filename)
        t_image, t_faces = detect_faces(frame_path)
        output_image = t_image.copy()

        if len(t_faces) > 0:
            target_face = t_faces[0]
            swapped = swap_faces(source_face, output_image, target_face)
            if first_target_crop is not None:
                swapped = match_histogram(swapped, first_target_crop)
            output_image = swapped

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, output_image)
        
        # Update progress dict
        progress_dict[output_dir] = (idx + 1) / total_frames  # Update progress as a fraction

def run_parallel_swap(frames_dir, base_output="swapped_vid_frame"):
    s_img, s_face = detect_faces("/content/h2.png")
    if len(s_face) == 0:
        print("❌ Tidak ada wajah sumber.")
        return
    source_face = s_face[0]
    first_target_crop = None

    all_frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    mid = len(all_frames) // 2
    part1, part2 = all_frames[:mid], all_frames[mid:]

    # Shared Manager dictionary for progress tracking
    with Manager() as manager:
        progress_dict = manager.dict()

        # Create processes
        p1 = Process(target=swap_faces_on_frames_partial, args=(part1, source_face, first_target_crop, frames_dir, f"{base_output}_part1", progress_dict))
        p2 = Process(target=swap_faces_on_frames_partial, args=(part2, source_face, first_target_crop, frames_dir, f"{base_output}_part2", progress_dict))
        
        # Start the processes
        p1.start()
        p2.start()

        # Track progress with tqdm
        while p1.is_alive() or p2.is_alive():
            # Calculate progress from both parts
            p1_progress = progress_dict.get(f"{base_output}_part1", 0)
            p2_progress = progress_dict.get(f"{base_output}_part2", 0)
            total_progress = (p1_progress + p2_progress) / 2

            tqdm.write(f"Progress: {total_progress * 100:.2f}%")
            tqdm.write(f"Part 1 Progress: {p1_progress * 100:.2f}% | Part 2 Progress: {p2_progress * 100:.2f}%")
            tqdm.write("-" * 50)

            time.sleep(1)  # Update progress every second

        # Wait for processes to finish
        p1.join()
        p2.join()

    print("🧩 Menggabungkan hasil frame...")
    final_output = base_output
    os.makedirs(final_output, exist_ok=True)

    # Merge the result from both parts
    for part_dir in [f"{base_output}_part1", f"{base_output}_part2"]:
        for fname in sorted(os.listdir(part_dir)):
            shutil.move(os.path.join(part_dir, fname), os.path.join(final_output, fname))
        os.rmdir(part_dir)

    print("✅ Proses selesai!")

def save_image(image):
    cv2.imwrite("swapped.jpg", image)

# Menjalankan script utama
if __name__ == "__main__":
  if initialize():
      # extract_video_frames(video_path="/content/haha.mp4")
        swap_faces_on_frames("vid_frame", base_output="swapped_vid_frame")
        create_video_from_frames("swapped_vid_frame", "output_video.mp4", fps=30)

      # s_img, s_face = detect_faces("assets/h2.png")
      # t_image, t_face = detect_faces("assets/hh.jpg")
      # print(f"Deteksi wajah: {len(s_face)} wajah ditemukan.")
      # swp = swap_faces(source_face= s_face[0], target_image= t_image, target_face= t_face[0])
      # # Ambil area wajah target
      # src_crop = extract_face(s_face[0], s_img)
      # target_crop = extract_face(t_face[0], t_image)
      # cv2.imwrite("cropped_src.jpg", src_crop)
      # cv2.imwrite("cropped_target.jpg", target_crop)
      # # Color correction
      # swp_corrected = match_histogram(swp, target_crop)

      # save_image(swp_corrected)

