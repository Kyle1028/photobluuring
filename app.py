import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, abort, url_for
from flask_sqlalchemy import SQLAlchemy

try:
    # MediaPipe
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
    MP_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
# 檔案存放
UPLOAD_IMAGE_DIR = BASE_DIR / "uploads" / "images"
UPLOAD_VIDEO_DIR = BASE_DIR / "uploads" / "videos"
OUTPUT_IMAGE_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"
PREVIEW_DIR = BASE_DIR / "previews"
METADATA_DIR = BASE_DIR / "metadata"
MODEL_DIR = BASE_DIR / "models"

for d in (UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, PREVIEW_DIR, METADATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# 資料庫設定
DATABASE_PATH = BASE_DIR / "database" / "app.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# 資料庫模型
class Media(db.Model):
    """媒體檔案記錄"""
    __tablename__ = "media"
    
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    original_filename = db.Column(db.String(255))  # 原始檔名
    file_type = db.Column(db.String(10), nullable=False)  # image / video
    upload_path = db.Column(db.String(500))  # 上傳檔案路徑
    output_path = db.Column(db.String(500))  # 處理後檔案路徑
    process_mode = db.Column(db.String(20))  # mosaic / eyes / replace
    face_count = db.Column(db.Integer, default=0)  # 偵測到的人臉數量
    status = db.Column(db.String(20), default="uploaded")  # uploaded / processed
    created_at = db.Column(db.DateTime, default=datetime.now)
    processed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f"<Media {self.media_id}>"


# 建立資料表
with app.app_context():
    db.create_all()

FACE_CASCADE = None
EYE_CASCADE = None

FACE_LANDMARKER_IMAGE = None
if MP_AVAILABLE:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "face_landmarker.task"
        if not model_path.exists():
            import urllib.request

            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options_image = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        # 建立 IMAGE 模式
        FACE_LANDMARKER_IMAGE = mp_vision.FaceLandmarker.create_from_options(options_image)
    except Exception:
        FACE_LANDMARKER_IMAGE = None


def _create_face_landmarker_video():
    # 每次處理影片建立新的 VIDEO landmarker，避免時間戳錯誤
    if not MP_AVAILABLE:
        return None
    try:
        model_path = MODEL_DIR / "face_landmarker.task"
        if not model_path.exists():
            return None
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options_video = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        return mp_vision.FaceLandmarker.create_from_options(options_video)
    except Exception:
        return None

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".webm"}


def _generate_media_id() -> str:
    """產生具有時間戳的 media_id，格式：YYYYMMDD_HHMMSS_shortid"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_id}"


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXT


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_VIDEO_EXT


def _detect_landmarks_bgr(
    image_bgr: np.ndarray,
    landmarker,
    timestamp_ms: int | None = None,
):
    # 回傳landmarks + 人臉框
    if landmarker is None:
        return [], np.array([])
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        mp_result = landmarker.detect(mp_image)
    else:
        mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not mp_result.face_landmarks:
        return [], np.array([])
    h_img, w_img = image_bgr.shape[:2]
    boxes = []
    for landmarks in mp_result.face_landmarks:
        xs = [int(lm.x * w_img) for lm in landmarks]
        ys = [int(lm.y * h_img) for lm in landmarks]
        if not xs or not ys:
            continue
        min_x, max_x = max(0, min(xs)), min(w_img, max(xs))
        min_y, max_y = max(0, min(ys)), min(h_img, max(ys))
        w = max_x - min_x
        h = max_y - min_y
        if w > 0 and h > 0:
            boxes.append((min_x, min_y, w, h))
    keep_idx = _nms_indices(boxes)
    if not keep_idx:
        return [], np.array([])
    filtered_landmarks = [mp_result.face_landmarks[i] for i in keep_idx]
    filtered_boxes = [boxes[i] for i in keep_idx]
    return filtered_landmarks, np.array(filtered_boxes)


def detect_faces_bgr(image_bgr: np.ndarray):
    # 取得人臉框
    if FACE_LANDMARKER_IMAGE is None:
        return np.array([])
    _, boxes = _detect_landmarks_bgr(image_bgr, FACE_LANDMARKER_IMAGE, timestamp_ms=None)
    return boxes


def _select_primary_face(faces):
    if faces is None or len(faces) == 0:
        return faces
    largest = max(faces, key=lambda f: f[2] * f[3])
    return np.array([largest])


def _sort_faces(faces):
    if faces is None or len(faces) == 0:
        return faces
    return np.array(sorted(faces, key=lambda f: (f[0], f[1])))


def _smooth_faces(prev_faces, curr_faces, alpha=0.7):
    # 平滑人臉框避免抖動
    if curr_faces is None or len(curr_faces) == 0:
        return prev_faces
    if prev_faces is None or len(prev_faces) == 0:
        return curr_faces
    prev_faces = _sort_faces(prev_faces)
    curr_faces = _sort_faces(curr_faces)
    if len(prev_faces) != len(curr_faces):
        return curr_faces
    smoothed = []
    for (px, py, pw, ph), (cx, cy, cw, ch) in zip(prev_faces, curr_faces):
        sx = int(round(alpha * px + (1 - alpha) * cx))
        sy = int(round(alpha * py + (1 - alpha) * cy))
        sw = int(round(alpha * pw + (1 - alpha) * cw))
        sh = int(round(alpha * ph + (1 - alpha) * ch))
        smoothed.append((sx, sy, sw, sh))
    return np.array(smoothed)


def _compute_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter_area / float(area_a + area_b - inter_area)


def _nms_indices(faces, iou_thresh=0.45):
    # 去除重疊的人臉框
    if faces is None or len(faces) == 0:
        return []
    faces = np.array(faces)
    scores = faces[:, 2] * faces[:, 3]
    order = scores.argsort()[::-1]
    keep_idx = []
    while order.size > 0:
        i = order[0]
        keep_idx.append(int(i))
        if order.size == 1:
            break
        rest = []
        for j in order[1:]:
            if _compute_iou(faces[i], faces[j]) <= iou_thresh:
                rest.append(j)
        order = np.array(rest, dtype=int)
    return keep_idx


def _save_faces_metadata(image_bgr: np.ndarray, faces, media_id: str):
    # 產生人臉縮圖讓使用者選擇
    items = []
    if faces is None or len(faces) == 0:
        faces = []
    for idx, (x, y, w, h) in enumerate(faces):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = max(0, x + w), max(0, y + h)
        crop = image_bgr[y1:y2, x1:x2]
        crop_name = f"{media_id}_face_{idx}.jpg"
        crop_path = PREVIEW_DIR / crop_name
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        items.append(
            {"id": idx, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "file": crop_name}
        )
    meta_path = METADATA_DIR / f"{media_id}_faces.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f)
    return items


def _load_faces_metadata(media_id: str):
    meta_path = METADATA_DIR / f"{media_id}_faces.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_faces_by_indices(faces, selected_ids):
    # 只保留使用者選的人臉框
    if faces is None or len(faces) == 0:
        return faces
    if not selected_ids:
        return faces
    selected = []
    for idx in selected_ids:
        if 0 <= idx < len(faces):
            selected.append(tuple(faces[idx]))
    return np.array(selected)


def _filter_landmarks_by_indices(landmarks, selected_ids):
    # 只保留使用者選的人臉 landmarks
    if not landmarks:
        return []
    if not selected_ids:
        return landmarks
    selected = []
    for idx in selected_ids:
        if 0 <= idx < len(landmarks):
            selected.append(landmarks[idx])
    return selected


def draw_face_boxes(image_bgr: np.ndarray, faces):
    boxed = image_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return boxed


def apply_mosaic(image_bgr: np.ndarray, faces, block_size=12):
    # 馬賽克整張人臉
    result = image_bgr.copy()
    for (x, y, w, h) in faces:
        roi = result[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)))
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result[y : y + h, x : x + w] = mosaic
    return result


def _adaptive_alpha(prev_boxes, curr_boxes, base=0.5, min_alpha=0.2):
    if not prev_boxes or not curr_boxes:
        return base
    prev_boxes = sorted(prev_boxes, key=lambda b: (b[0], b[1]))
    curr_boxes = sorted(curr_boxes, key=lambda b: (b[0], b[1]))
    if len(prev_boxes) != len(curr_boxes):
        return base
    max_move = 0
    for (px1, py1, px2, py2), (cx1, cy1, cx2, cy2) in zip(prev_boxes, curr_boxes):
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        ccx = (cx1 + cx2) / 2.0
        ccy = (cy1 + cy2) / 2.0
        max_move = max(max_move, abs(ccx - pcx) + abs(ccy - pcy))
    if max_move > 30:
        return min_alpha
    if max_move > 12:
        return 0.3
    return base


def _smooth_boxes(prev_boxes, curr_boxes, alpha=0.5):
    if not curr_boxes:
        return prev_boxes or []
    if not prev_boxes:
        return curr_boxes
    prev_boxes = sorted(prev_boxes, key=lambda b: (b[0], b[1]))
    curr_boxes = sorted(curr_boxes, key=lambda b: (b[0], b[1]))
    if len(prev_boxes) != len(curr_boxes):
        return curr_boxes
    smoothed = []
    for (px1, py1, px2, py2), (cx1, cy1, cx2, cy2) in zip(prev_boxes, curr_boxes):
        sx1 = int(round(alpha * px1 + (1 - alpha) * cx1))
        sy1 = int(round(alpha * py1 + (1 - alpha) * cy1))
        sx2 = int(round(alpha * px2 + (1 - alpha) * cx2))
        sy2 = int(round(alpha * py2 + (1 - alpha) * cy2))
        smoothed.append((sx1, sy1, sx2, sy2))
    return smoothed


def _get_points(landmarks, idxs, w_img, h_img):
    # 取出指定特徵點座標
    points = []
    for idx in idxs:
        lm = landmarks[idx]
        points.append((int(lm.x * w_img), int(lm.y * h_img)))
    return points


def apply_eye_cover(
    image_bgr: np.ndarray,
    face_landmarks,
    prev_boxes=None,
):
    # 以眼睛特徵點做遮蓋
    result = image_bgr.copy()
    if not face_landmarks:
        if prev_boxes:
            for (x1, y1, x2, y2) in prev_boxes:
                cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)
            return result, prev_boxes
        return result, prev_boxes or []

    left_eye = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye = [362, 263, 387, 386, 385, 373, 374, 380]
    left_iris = [474, 475, 476, 477]
    right_iris = [469, 470, 471, 472]
    h_img, w_img = result.shape[:2]
    eye_boxes = []
    for landmarks in face_landmarks:
        left_iris_pts = _get_points(landmarks, left_iris, w_img, h_img)
        right_iris_pts = _get_points(landmarks, right_iris, w_img, h_img)
        left_eye_pts = _get_points(landmarks, left_eye, w_img, h_img)
        right_eye_pts = _get_points(landmarks, right_eye, w_img, h_img)

        if left_iris_pts and right_iris_pts:
            lx = int(sum(p[0] for p in left_iris_pts) / len(left_iris_pts))
            ly = int(sum(p[1] for p in left_iris_pts) / len(left_iris_pts))
            rx = int(sum(p[0] for p in right_iris_pts) / len(right_iris_pts))
            ry = int(sum(p[1] for p in right_iris_pts) / len(right_iris_pts))
        else:
            all_left = left_eye_pts
            all_right = right_eye_pts
            if not all_left or not all_right:
                continue
            lx = int(sum(p[0] for p in all_left) / len(all_left))
            ly = int(sum(p[1] for p in all_left) / len(all_left))
            rx = int(sum(p[0] for p in all_right) / len(all_right))
            ry = int(sum(p[1] for p in all_right) / len(all_right))

        center_x = int((lx + rx) / 2)
        center_y = int((ly + ry) / 2)
        eye_dist = max(12, int(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5))
        band_w = int(eye_dist * 1.8)
        band_h = int(eye_dist * 0.55)
        cover_x1 = max(0, center_x - band_w // 2)
        cover_x2 = min(w_img, center_x + band_w // 2)
        cover_y1 = max(0, center_y - band_h // 2)
        cover_y2 = min(h_img, center_y + band_h // 2)
        eye_boxes.append((cover_x1, cover_y1, cover_x2, cover_y2))

    alpha = _adaptive_alpha(prev_boxes or [], eye_boxes, base=0.45, min_alpha=0.2)
    eye_boxes = _smooth_boxes(prev_boxes, eye_boxes, alpha=alpha)
    for (x1, y1, x2, y2) in eye_boxes:
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return result, eye_boxes


def _load_overlay_rgba(path: Path):
    # 讀取遮擋圖片
    overlay = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return None
    if overlay.shape[2] == 3:
        alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=np.uint8) * 255
        overlay = np.concatenate([overlay, alpha], axis=2)
    return overlay


def apply_face_replace(image_bgr: np.ndarray, faces, overlay_rgba: np.ndarray):
    # 以遮擋圖片覆蓋人臉
    result = image_bgr.copy()
    for (x, y, w, h) in faces:
        if w <= 0 or h <= 0:
            continue
        overlay_resized = cv2.resize(overlay_rgba, (w, h))
        overlay_rgb = overlay_resized[:, :, :3]
        alpha = overlay_resized[:, :, 3:4] / 255.0
        roi = result[y : y + h, x : x + w]
        if roi.shape[:2] != overlay_rgb.shape[:2]:
            continue
        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        result[y : y + h, x : x + w] = blended
    return result

# 開啟影片寫入器
def _open_video_writer(out_base: Path, fps: float, size: tuple[int, int]):
    # 依序嘗試可用的影片編碼器

    candidates = [("avc1", ".mp4"), ("mp4v", ".mp4"), ("VP80", ".webm")]
    for fourcc_name, ext in candidates:
        out_path = out_base.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if writer.isOpened():
            return writer, out_path
    return None, None


def _save_preview(image_bgr: np.ndarray, name: str):
    preview_path = PREVIEW_DIR / f"{name}.jpg"
    cv2.imwrite(str(preview_path), image_bgr)
    return preview_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # 上傳並進行人臉標註預覽
    file = request.files.get("media")
    if not file or not file.filename:
        abort(400, "未提供檔案")
    original_filename = file.filename
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
        abort(400, "檔案格式不支援")

    media_id = _generate_media_id()
    # 根據檔案類型存到不同資料夾
    if ext in ALLOWED_IMAGE_EXT:
        saved_path = UPLOAD_IMAGE_DIR / f"{media_id}{ext}"
        file_type = "image"
    else:
        saved_path = UPLOAD_VIDEO_DIR / f"{media_id}{ext}"
        file_type = "video"
    file.save(saved_path)

    if _is_image(saved_path):
        image = cv2.imread(str(saved_path))
        _, faces = _detect_landmarks_bgr(image, FACE_LANDMARKER_IMAGE, None)
        faces_info = _save_faces_metadata(image, faces, media_id)
        preview = draw_face_boxes(image, faces)
        preview_path = _save_preview(preview, media_id)
        
        # 記錄到資料庫
        media_record = Media(
            media_id=media_id,
            original_filename=original_filename,
            file_type=file_type,
            upload_path=str(saved_path),
            face_count=len(faces_info),
            status="uploaded",
        )
        db.session.add(media_record)
        db.session.commit()
        
        return render_template(
            "options.html",
            media_id=media_id,
            is_video=False,
            preview_url=url_for("previews", filename=preview_path.name),
            faces=faces_info,
        )

    # 抓第一幀
    cap = cv2.VideoCapture(str(saved_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        abort(400, "無法讀取影片")
    _, faces = _detect_landmarks_bgr(frame, FACE_LANDMARKER_IMAGE, None)
    faces_info = _save_faces_metadata(frame, faces, media_id)
    preview = draw_face_boxes(frame, faces)
    preview_path = _save_preview(preview, media_id)
    
    # 記錄到資料庫
    media_record = Media(
        media_id=media_id,
        original_filename=original_filename,
        file_type=file_type,
        upload_path=str(saved_path),
        face_count=len(faces_info),
        status="uploaded",
    )
    db.session.add(media_record)
    db.session.commit()
    
    return render_template(
        "options.html",
        media_id=media_id,
        is_video=True,
        preview_url=url_for("previews", filename=preview_path.name),
        faces=faces_info,
    )


@app.route("/process", methods=["POST"])
def process():
    # 依選擇的模式處理照片/影片
    media_id = request.form.get("media_id", "").strip()
    mode = request.form.get("mode", "").strip()
    face_ids_raw = request.form.getlist("face_ids")
    selected_ids = []
    for item in face_ids_raw:
        try:
            selected_ids.append(int(item))
        except ValueError:
            continue
    if not media_id:
        abort(400, "缺少 media_id")

    # 在圖片和影片資料夾中搜尋檔案
    candidates = list(UPLOAD_IMAGE_DIR.glob(f"{media_id}.*"))
    if not candidates:
        candidates = list(UPLOAD_VIDEO_DIR.glob(f"{media_id}.*"))
    if not candidates:
        abort(404, "找不到檔案")
    src_path = candidates[0]

    overlay_file = request.files.get("overlay")
    overlay_path = None
    if overlay_file and overlay_file.filename:
        overlay_ext = Path(overlay_file.filename).suffix.lower()
        if overlay_ext not in ALLOWED_IMAGE_EXT:
            abort(400, "圖片格式不支援")
        overlay_path = UPLOAD_IMAGE_DIR / f"{media_id}_overlay{overlay_ext}"
        overlay_file.save(overlay_path)

    if _is_image(src_path):
        image = cv2.imread(str(src_path))
        face_landmarks, faces = _detect_landmarks_bgr(image, FACE_LANDMARKER_IMAGE, None)
        face_landmarks = _filter_landmarks_by_indices(face_landmarks, selected_ids)
        faces = _filter_faces_by_indices(faces, selected_ids)
        if mode == "mosaic":
            output = apply_mosaic(image, faces)
        elif mode == "eyes":
            output, _ = apply_eye_cover(image, face_landmarks, prev_boxes=None)
        elif mode == "replace":
            if overlay_path is None:
                abort(400, "請上傳替換圖片")
            overlay = _load_overlay_rgba(overlay_path)
            if overlay is None:
                abort(400, "替換圖片讀取失敗")
            output = apply_face_replace(image, faces, overlay)


        out_name = f"{media_id}_out.jpg"
        out_path = OUTPUT_IMAGE_DIR / out_name
        cv2.imwrite(str(out_path), output)
        
        # 更新資料庫
        media_record = Media.query.filter_by(media_id=media_id).first()
        if media_record:
            media_record.output_path = str(out_path)
            media_record.process_mode = mode
            media_record.status = "processed"
            media_record.processed_at = datetime.now()
            db.session.commit()
        
        return render_template(
            "result.html", is_video=False, result_url=url_for("output_images", filename=out_name)
        )

    # 影片處理
    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 24
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        abort(400, "無法讀取影片影格")
    height, width = first_frame.shape[:2]
    out_base = OUTPUT_VIDEO_DIR / f"{media_id}_out"
    writer, out_path = _open_video_writer(out_base, fps, (width, height))
    if writer is None:
        cap.release()
        abort(500, "影片編碼器初始失敗")

    overlay = None
    if mode == "replace":
        if overlay_path is None:
            abort(400, "請上傳替換圖片")
        overlay = _load_overlay_rgba(overlay_path)
        if overlay is None:
            abort(400, "替換圖片讀取失敗")

    # process first frame
    prev_faces = None
    prev_eye_boxes = []
    frame_idx = 0
    video_landmarker = _create_face_landmarker_video()
    if video_landmarker is None:
        cap.release()
        writer.release()
        abort(500, "MediaPipe 初始化失敗")
    face_landmarks, faces = _detect_landmarks_bgr(first_frame, video_landmarker, 0)
    faces = _smooth_faces(prev_faces, faces)
    prev_faces = faces
    face_landmarks = _filter_landmarks_by_indices(face_landmarks, selected_ids)
    faces = _filter_faces_by_indices(faces, selected_ids)
    if mode == "mosaic":
        processed = apply_mosaic(first_frame, faces)
    elif mode == "eyes":
        processed, prev_eye_boxes = apply_eye_cover(
            first_frame, face_landmarks, prev_eye_boxes
        )
    elif mode == "replace":
        processed = apply_face_replace(first_frame, faces, overlay)

    writer.write(processed)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        timestamp_ms = int(frame_idx * 1000 / fps)
        face_landmarks, faces = _detect_landmarks_bgr(
            frame, video_landmarker, timestamp_ms
        )
        faces = _smooth_faces(prev_faces, faces)
        prev_faces = faces
        face_landmarks = _filter_landmarks_by_indices(face_landmarks, selected_ids)
        faces = _filter_faces_by_indices(faces, selected_ids)
        if mode == "mosaic":
            processed = apply_mosaic(frame, faces)
        elif mode == "eyes":
            processed, prev_eye_boxes = apply_eye_cover(
                frame, face_landmarks, prev_eye_boxes
            )
        elif mode == "replace":
            processed = apply_face_replace(frame, faces, overlay)

        writer.write(processed)

    cap.release()
    writer.release()
    
    # 更新資料庫
    media_record = Media.query.filter_by(media_id=media_id).first()
    if media_record:
        media_record.output_path = str(out_path)
        media_record.process_mode = mode
        media_record.status = "processed"
        media_record.processed_at = datetime.now()
        db.session.commit()
    
    return render_template(
        "result.html",
        is_video=True,
        result_url=url_for("output_videos", filename=out_path.name),
    )


@app.route("/outputs/images/<path:filename>")
def output_images(filename):
    return send_from_directory(OUTPUT_IMAGE_DIR, filename, as_attachment=False)


@app.route("/outputs/videos/<path:filename>")
def output_videos(filename):
    return send_from_directory(OUTPUT_VIDEO_DIR, filename, as_attachment=False)


@app.route("/previews/<path:filename>")
def previews(filename):
    return send_from_directory(PREVIEW_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
