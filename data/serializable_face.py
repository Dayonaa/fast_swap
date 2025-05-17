from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import os
import cv2


@dataclass
class SerializableFace:
    bbox: List[float]
    kps: List[List[float]]
    pose: Optional[List[float]]
    embedding: Optional[List[float]]
    normed_embedding: Optional[List[float]]
    det_score: float
    gender: Optional[int] = None
    age: Optional[int] = None
    landmark_2d_106: Optional[List[List[float]]] = None
    aligned_path: Optional[str] = None
    image_id: Optional[str] = None

    @classmethod
    def from_face(
        cls,
        face,
        src_img: np.ndarray,
        aligned_save_dir: Optional[str] = None,
        image_id: Optional[str] = None,
    ):
        """
        Convert a raw Face object to SerializableFace dan simpan aligned face secara manual.
        """
        aligned_face = None
        if face.kps is not None:
            try:
                # Gunakan fungsi align bawaan InsightFace
                from insightface.utils import face_align

                aligned_face = face_align.norm_crop(src_img, face.kps, 224)
            except Exception as e:
                print(f"⚠️ Gagal crop aligned face: {e}")

        # Simpan jika ada dan diminta
        if aligned_save_dir and aligned_face is not None:
            os.makedirs(aligned_save_dir, exist_ok=True)
            save_path = os.path.join(aligned_save_dir, f"{image_id}.jpg")
            success = cv2.imwrite(save_path, aligned_face)
            if not success:
                print(f"⚠️ Gagal simpan aligned face ke {save_path}")

        return cls(
            bbox=face.bbox.tolist(),
            kps=face.kps.tolist(),
            pose=(face.pose if hasattr(face, "pose") else None),
            embedding=face.embedding.tolist() if hasattr(face, "embedding") else None,
            normed_embedding=(
                face.normed_embedding.tolist()
                if hasattr(face, "normed_embedding")
                else None
            ),
            det_score=face.det_score,
            gender=getattr(face, "gender", None),
            age=getattr(face, "age", None),
            landmark_2d_106=(
                face.landmark_2d_106.tolist()
                if hasattr(face, "landmark_2d_106")
                else None
            ),
            image_id=image_id,
        )

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "kps": self.kps,
            "pose": self.pose,
            "embedding": self.embedding,
            "normed_embedding": self.normed_embedding,
            "det_score": self.det_score,
            "gender": self.gender,
            "age": self.age,
            "landmark_2d_106": self.landmark_2d_106,
            "aligned_path": self.aligned_path,
            "image_id": self.image_id,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
