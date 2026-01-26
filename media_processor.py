"""
媒體處理模組：照片和影片的人臉隱私處理
提供統一的處理介面，可被其他模組調用
"""
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np


class MediaProcessor:
    """
    媒體處理器類別
    提供照片和影片的統一處理介面
    """
    
    def __init__(self, sensitivity: float = 0.6):
        """
        初始化處理器
        
        參數:
            sensitivity: 人臉偵測靈敏度 (0.3-0.9)
        """
        self.sensitivity = max(0.3, min(0.9, sensitivity))
        self.image_landmarker = None
        self.video_landmarker = None
        self._app_funcs = None
    
    def _get_app_funcs(self):
        """取得 app.py 中的函數（延遲載入，避免循環導入）"""
        if self._app_funcs is None:
            # 延遲導入，避免循環導入問題
            from app import (
                _detect_landmarks_bgr,
                _filter_landmarks_by_indices,
                _filter_faces_by_indices,
                apply_mosaic,
                apply_eye_cover,
                apply_face_replace,
                _load_overlay_rgba,
                _smooth_faces,
                _create_face_landmarker_image,
                _create_face_landmarker_video,
                _open_video_writer,
                _is_image,
                _is_video,
                OUTPUT_IMAGE_DIR,
                OUTPUT_VIDEO_DIR,
            )
            self._app_funcs = {
                '_detect_landmarks_bgr': _detect_landmarks_bgr,
                '_filter_landmarks_by_indices': _filter_landmarks_by_indices,
                '_filter_faces_by_indices': _filter_faces_by_indices,
                'apply_mosaic': apply_mosaic,
                'apply_eye_cover': apply_eye_cover,
                'apply_face_replace': apply_face_replace,
                '_load_overlay_rgba': _load_overlay_rgba,
                '_smooth_faces': _smooth_faces,
                '_create_face_landmarker_image': _create_face_landmarker_image,
                '_create_face_landmarker_video': _create_face_landmarker_video,
                '_open_video_writer': _open_video_writer,
                '_is_image': _is_image,
                '_is_video': _is_video,
                'OUTPUT_IMAGE_DIR': OUTPUT_IMAGE_DIR,
                'OUTPUT_VIDEO_DIR': OUTPUT_VIDEO_DIR,
            }
        return self._app_funcs
    
    def _get_image_landmarker(self):
        """取得照片用的人臉偵測器（延遲初始化）"""
        if self.image_landmarker is None:
            funcs = self._get_app_funcs()
            self.image_landmarker = funcs['_create_face_landmarker_image'](self.sensitivity)
        return self.image_landmarker
    
    def _get_video_landmarker(self):
        """取得影片用的人臉偵測器（延遲初始化）"""
        if self.video_landmarker is None:
            funcs = self._get_app_funcs()
            self.video_landmarker = funcs['_create_face_landmarker_video'](self.sensitivity)
        return self.video_landmarker
    
    def process_image(
        self,
        image_path: Path,
        mode: str,
        selected_face_ids: Optional[List[int]] = None,
        overlay_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        處理照片
        
        參數:
            image_path: 輸入照片路徑
            mode: 處理模式 ('mosaic', 'eyes', 'replace')
            selected_face_ids: 要處理的人臉 ID 列表（None 表示處理所有人臉）
            overlay_path: 替換模式用的覆蓋圖片路徑
            output_path: 輸出檔案路徑（None 時自動產生）
        
        回傳:
            輸出檔案路徑
        
        範例:
            processor = MediaProcessor(sensitivity=0.6)
            output = processor.process_image(
                image_path=Path("input.jpg"),
                mode="mosaic",
                selected_face_ids=[0, 1],  # 只處理前兩張人臉
            )
        """
        funcs = self._get_app_funcs()
        _is_image = funcs['_is_image']
        _detect_landmarks_bgr = funcs['_detect_landmarks_bgr']
        _filter_landmarks_by_indices = funcs['_filter_landmarks_by_indices']
        _filter_faces_by_indices = funcs['_filter_faces_by_indices']
        apply_mosaic = funcs['apply_mosaic']
        apply_eye_cover = funcs['apply_eye_cover']
        apply_face_replace = funcs['apply_face_replace']
        _load_overlay_rgba = funcs['_load_overlay_rgba']
        OUTPUT_IMAGE_DIR = funcs['OUTPUT_IMAGE_DIR']
        
        if not _is_image(image_path):
            raise ValueError(f"不支援的圖片格式: {image_path}")
        
        if mode not in ['mosaic', 'eyes', 'replace']:
            raise ValueError(f"不支援的處理模式: {mode}")
        
        # 載入圖片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
        
        # 偵測人臉
        landmarker = self._get_image_landmarker()
        if landmarker is None:
            raise RuntimeError("無法初始化人臉偵測器")
        
        face_landmarks, faces = _detect_landmarks_bgr(image, landmarker, None)
        
        # 篩選要處理的人臉
        if selected_face_ids is not None:
            face_landmarks = _filter_landmarks_by_indices(face_landmarks, selected_face_ids)
            faces = _filter_faces_by_indices(faces, selected_face_ids)
        
        # 根據模式處理
        if mode == "mosaic":
            output = apply_mosaic(image, faces)
        elif mode == "eyes":
            output, _ = apply_eye_cover(image, face_landmarks, prev_boxes=None)
        elif mode == "replace":
            if overlay_path is None:
                raise ValueError("替換模式需要提供 overlay_path")
            overlay = _load_overlay_rgba(overlay_path)
            if overlay is None:
                raise ValueError(f"無法讀取覆蓋圖片: {overlay_path}")
            output = apply_face_replace(image, faces, overlay)
        
        # 儲存結果
        if output_path is None:
            output_path = OUTPUT_IMAGE_DIR / f"{image_path.stem}_processed.jpg"
        
        OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)
        
        return output_path
    
    def process_video(
        self,
        video_path: Path,
        mode: str,
        selected_face_ids: Optional[List[int]] = None,
        overlay_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        處理影片
        
        參數:
            video_path: 輸入影片路徑
            mode: 處理模式 ('mosaic', 'eyes', 'replace')
            selected_face_ids: 要處理的人臉 ID 列表（None 表示處理所有人臉）
            overlay_path: 替換模式用的覆蓋圖片路徑
            output_path: 輸出檔案路徑（None 時自動產生）
        
        回傳:
            輸出檔案路徑
        
        範例:
            processor = MediaProcessor(sensitivity=0.6)
            output = processor.process_video(
                video_path=Path("input.mp4"),
                mode="mosaic",
                selected_face_ids=[0],  # 只處理第一張人臉
            )
        """
        funcs = self._get_app_funcs()
        _is_video = funcs['_is_video']
        _detect_landmarks_bgr = funcs['_detect_landmarks_bgr']
        _filter_landmarks_by_indices = funcs['_filter_landmarks_by_indices']
        _filter_faces_by_indices = funcs['_filter_faces_by_indices']
        apply_mosaic = funcs['apply_mosaic']
        apply_eye_cover = funcs['apply_eye_cover']
        apply_face_replace = funcs['apply_face_replace']
        _load_overlay_rgba = funcs['_load_overlay_rgba']
        _smooth_faces = funcs['_smooth_faces']
        _open_video_writer = funcs['_open_video_writer']
        OUTPUT_VIDEO_DIR = funcs['OUTPUT_VIDEO_DIR']
        
        if not _is_video(video_path):
            raise ValueError(f"不支援的影片格式: {video_path}")
        
        if mode not in ['mosaic', 'eyes', 'replace']:
            raise ValueError(f"不支援的處理模式: {mode}")
        
        # 開啟影片
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        # 取得影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = 24
        
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError("無法讀取影片影格")
        
        height, width = first_frame.shape[:2]
        
        # 建立輸出影片寫入器
        if output_path is None:
            output_base = OUTPUT_VIDEO_DIR / f"{video_path.stem}_processed"
        else:
            output_base = output_path.with_suffix('')
        
        OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        writer, out_path = _open_video_writer(output_base, fps, (width, height))
        if writer is None:
            cap.release()
            raise RuntimeError("無法初始化影片編碼器")
        
        # 載入覆蓋圖片（如果需要）
        overlay = None
        if mode == "replace":
            if overlay_path is None:
                cap.release()
                writer.release()
                raise ValueError("替換模式需要提供 overlay_path")
            overlay = _load_overlay_rgba(overlay_path)
            if overlay is None:
                cap.release()
                writer.release()
                raise ValueError(f"無法讀取覆蓋圖片: {overlay_path}")
        
        # 初始化人臉偵測器
        landmarker = self._get_video_landmarker()
        if landmarker is None:
            cap.release()
            writer.release()
            raise RuntimeError("無法初始化人臉偵測器")
        
        # 處理第一幀
        prev_faces = None
        prev_eye_boxes = []
        frame_idx = 0
        
        face_landmarks, faces = _detect_landmarks_bgr(first_frame, landmarker, 0)
        faces = _smooth_faces(prev_faces, faces)
        prev_faces = faces
        
        if selected_face_ids is not None:
            face_landmarks = _filter_landmarks_by_indices(face_landmarks, selected_face_ids)
            faces = _filter_faces_by_indices(faces, selected_face_ids)
        
        if mode == "mosaic":
            processed = apply_mosaic(first_frame, faces)
        elif mode == "eyes":
            processed, prev_eye_boxes = apply_eye_cover(
                first_frame, face_landmarks, prev_eye_boxes
            )
        elif mode == "replace":
            processed = apply_face_replace(first_frame, faces, overlay)
        
        writer.write(processed)
        
        # 處理後續影格
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_idx += 1
            timestamp_ms = int(frame_idx * 1000 / fps)
            
            # 偵測人臉
            face_landmarks, faces = _detect_landmarks_bgr(
                frame, landmarker, timestamp_ms
            )
            faces = _smooth_faces(prev_faces, faces)
            prev_faces = faces
            
            # 篩選要處理的人臉
            if selected_face_ids is not None:
                face_landmarks = _filter_landmarks_by_indices(face_landmarks, selected_face_ids)
                faces = _filter_faces_by_indices(faces, selected_face_ids)
            
            # 根據模式處理
            if mode == "mosaic":
                processed = apply_mosaic(frame, faces)
            elif mode == "eyes":
                processed, prev_eye_boxes = apply_eye_cover(
                    frame, face_landmarks, prev_eye_boxes
                )
            elif mode == "replace":
                processed = apply_face_replace(frame, faces, overlay)
            
            writer.write(processed)
        
        # 清理資源
        cap.release()
        writer.release()
        
        return out_path
    
    def process(
        self,
        media_path: Path,
        mode: str,
        selected_face_ids: Optional[List[int]] = None,
        overlay_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        統一處理介面（自動判斷照片或影片）
        
        參數:
            media_path: 輸入媒體檔案路徑
            mode: 處理模式 ('mosaic', 'eyes', 'replace')
            selected_face_ids: 要處理的人臉 ID 列表（None 表示處理所有人臉）
            overlay_path: 替換模式用的覆蓋圖片路徑
            output_path: 輸出檔案路徑（None 時自動產生）
        
        回傳:
            輸出檔案路徑
        
        範例:
            processor = MediaProcessor(sensitivity=0.6)
            output = processor.process(
                media_path=Path("input.jpg"),
                mode="mosaic",
            )
        """
        funcs = self._get_app_funcs()
        _is_image = funcs['_is_image']
        _is_video = funcs['_is_video']
        
        if _is_image(media_path):
            return self.process_image(
                media_path, mode, selected_face_ids, overlay_path, output_path
            )
        elif _is_video(media_path):
            return self.process_video(
                media_path, mode, selected_face_ids, overlay_path, output_path
            )
        else:
            raise ValueError(f"不支援的媒體格式: {media_path}")


# 便利函數：快速處理
def process_media(
    media_path: Path,
    mode: str,
    sensitivity: float = 0.6,
    selected_face_ids: Optional[List[int]] = None,
    overlay_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    快速處理媒體檔案的便利函數
    
    參數:
        media_path: 輸入媒體檔案路徑
        mode: 處理模式 ('mosaic', 'eyes', 'replace')
        sensitivity: 人臉偵測靈敏度 (0.3-0.9)
        selected_face_ids: 要處理的人臉 ID 列表
        overlay_path: 替換模式用的覆蓋圖片路徑
        output_path: 輸出檔案路徑
    
    回傳:
        輸出檔案路徑
    
    範例:
        output = process_media(
            media_path=Path("input.jpg"),
            mode="mosaic",
            sensitivity=0.7,
            selected_face_ids=[0, 1],
        )
    """
    processor = MediaProcessor(sensitivity=sensitivity)
    return processor.process(
        media_path, mode, selected_face_ids, overlay_path, output_path
    )
