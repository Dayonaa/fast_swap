import argparse
from face_util_kag import FaceUtil
from video_util_kag import VideoUtil

def main():
    parser = argparse.ArgumentParser(description="Fast Swap Tool")
    parser.add_argument('--vid-path', type=str, required=True, help='Path ke video input')
    parser.add_argument('--src-img-path', type=str, required=True, help='Path ke gambar sumber wajah')
    args = parser.parse_args()

    face_util = FaceUtil()
    video_util = VideoUtil()


    video_util.extract_video_frames(args.vid_path)

    video_util.swap_video(args.src_img_path)

    video_util.create_video_from_frames(fps=30)

if __name__ == "__main__":
    main()
