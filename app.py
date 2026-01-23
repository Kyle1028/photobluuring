"""
主程式檔案：處理照片/影片上傳、人臉偵測、隱私處理
"""
# ==================== 套件匯入 ====================
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2  # OpenCV：影像處理
import numpy as np  # NumPy：數值運算
from flask import Flask, render_template, request, send_from_directory, abort, url_for, redirect, session
from flask_login import login_required, current_user
from flask_babel import Babel, gettext, lazy_gettext

from auth import init_auth  # 認證系統
from models import db, Media  # 資料庫模型

# 匯入 MediaPipe（用於人臉偵測）
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True  # MediaPipe 可用
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
    MP_AVAILABLE = False  # MediaPipe 不可用（會影響人臉偵測功能）


# ==================== 目錄設定 ====================
BASE_DIR = Path(__file__).resolve().parent  # 專案根目錄

# 各類檔案的儲存目錄
UPLOAD_IMAGE_DIR = BASE_DIR / "uploads" / "images"    # 上傳的照片
UPLOAD_VIDEO_DIR = BASE_DIR / "uploads" / "videos"    # 上傳的影片
OUTPUT_IMAGE_DIR = BASE_DIR / "outputs" / "images"    # 處理後的照片
OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"    # 處理後的影片
PREVIEW_DIR = BASE_DIR / "previews"                    # 預覽圖（含人臉框）
METADATA_DIR = BASE_DIR / "metadata"                   # 人臉資料（JSON）
MODEL_DIR = BASE_DIR / "models"                        # AI 模型檔案

# 自動建立所有需要的目錄（如果不存在）
for d in (UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, PREVIEW_DIR, METADATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ==================== Flask 應用程式設定 ====================
app = Flask(__name__)

# 資料庫設定 - 從環境變數讀取連線資訊
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "缺少 DATABASE_URL 環境變數！\n"
        "請設定: set DATABASE_URL=mysql+pymysql://root:password@localhost:3306/photobluuring"
    )

# 資料庫配置
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False  # 關閉追蹤修改（節省資源）
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,      # 連線前檢查是否有效
    "pool_recycle": 3600,        # 每小時回收連線（避免逾時）
}

# Session 密鑰（用於加密 Cookie）
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-please-change-in-production")

# 多語言設定
app.config["BABEL_DEFAULT_LOCALE"] = "zh_Hant_TW"  # 預設語言：繁體中文
app.config["BABEL_TRANSLATION_DIRECTORIES"] = "translations"  # 翻譯檔案目錄
app.config["LANGUAGES"] = {
    "zh_Hant_TW": "繁體中文",
    "en": "English"
}

# 郵件配置（預留功能）
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")

# 初始化資料庫
db.init_app(app)

# 初始化認證系統
init_auth(app)

# 初始化 Babel（多語言）
babel = Babel(app)


def get_locale():
    """
    取得使用者的語言偏好
    優先順序：
    1. URL 參數 ?lang=en
    2. Session 中儲存的語言
    3. 預設語言（繁體中文）
    """
    # 如果 URL 有指定語言，儲存到 session
    if request.args.get('lang'):
        session['lang'] = request.args.get('lang')
    
    # 從 session 取得語言
    if 'lang' in session:
        return session['lang']
    
    # 回傳預設語言（繁體中文）
    return app.config['BABEL_DEFAULT_LOCALE']


# 設定 Babel 的語言選擇函式
babel.init_app(app, locale_selector=get_locale)

# 建立資料表（如果不存在）
with app.app_context():
    db.create_all()



# ==================== MediaPipe 人臉偵測器初始化 ====================

def _create_face_landmarker_image(min_detection_confidence=0.5):
    """
    建立照片用的人臉特徵偵測器
    
    參數：
        min_detection_confidence: 最低信心度（0.0-1.0），數值越低越容易偵測到人臉
    
    回傳：
        FaceLandmarker 物件（可偵測人臉並找出 478 個特徵點）
    """
    if not MP_AVAILABLE:
        return None
    
    try:
        # 確保模型目錄存在
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "face_landmarker.task"
        
        # 如果模型檔案不存在，自動從 Google 下載
        if not model_path.exists():
            import urllib.request
            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
        
        # 設定模型參數
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options_image = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,  # 最多偵測 5 張人臉
            min_face_detection_confidence=min_detection_confidence,  # 最低偵測信心度
            min_face_presence_confidence=min_detection_confidence,   # 最低存在信心度
            running_mode=mp_vision.RunningMode.IMAGE,  # 照片模式
        )
        return mp_vision.FaceLandmarker.create_from_options(options_image)
    except Exception:
        return None


# 建立預設的照片人臉偵測器（信心度 0.6）
FACE_LANDMARKER_IMAGE = _create_face_landmarker_image(0.6) if MP_AVAILABLE else None


def _create_face_landmarker_video(min_detection_confidence=0.5):
    """
    建立影片用的人臉特徵偵測器
    
    參數：
        min_detection_confidence: 最低信心度（0.3-0.9）
    
    回傳：
        FaceLandmarker 物件（影片模式，處理每一幀）
    """
    if not MP_AVAILABLE:
        return None
    
    try:
        model_path = MODEL_DIR / "face_landmarker.task"
        if not model_path.exists():
            return None
        
        # 設定模型參數（影片模式）
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options_video = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,  # 最多偵測 5 張人臉
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,  # 影片模式
        )
        return mp_vision.FaceLandmarker.create_from_options(options_video)
    except Exception:
        return None


# ==================== 檔案類型設定 ====================
# 允許的檔案格式
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".webm"}




def _generate_media_id() -> str:
    """
    產生唯一的媒體檔案 ID
    格式：YYYYMMDD_HHMMSS_隨機8碼
    例如：20260123_143052_a3b9f2c1
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]  # 取 UUID 前 8 碼
    return f"{timestamp}_{short_id}"


def _is_image(path: Path) -> bool:
    """
    判斷檔案是否為圖片
    """
    return path.suffix.lower() in ALLOWED_IMAGE_EXT


def _is_video(path: Path) -> bool:
    """
    判斷檔案是否為影片
    """
    return path.suffix.lower() in ALLOWED_VIDEO_EXT



def _detect_landmarks_bgr(
    image_bgr: np.ndarray,
    landmarker,
    timestamp_ms: int | None = None,
):
    """
    偵測人臉並找出特徵點（478個關鍵點）
    
    參數：
        image_bgr: OpenCV 圖片（BGR 格式）
        landmarker: MediaPipe 人臉偵測器
        timestamp_ms: 影片時間戳（毫秒），照片模式時為 None
    
    回傳：
        (landmarks, boxes) - 特徵點陣列和邊界框陣列
    """
    if landmarker is None:
        return [], np.array([])
    
    # 步驟 1：將 BGR 轉換為 RGB（MediaPipe 需要）
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # 步驟 2：執行偵測
    if timestamp_ms is None:
        mp_result = landmarker.detect(mp_image)  # 照片模式
    else:
        mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)  # 影片模式
    
    # 如果沒有偵測到人臉，回傳空陣列
    if not mp_result.face_landmarks:
        return [], np.array([])
    
    # 步驟 3：將特徵點轉換為邊界框（bounding box）
    h_img, w_img = image_bgr.shape[:2]  # 取得圖片尺寸
    boxes = []
    
    for landmarks in mp_result.face_landmarks:
        # 取得所有特徵點的 x, y 座標
        xs = [int(lm.x * w_img) for lm in landmarks]
        ys = [int(lm.y * h_img) for lm in landmarks]
        
        if not xs or not ys:
            continue
        
        # 計算最小的矩形框住所有特徵點
        min_x, max_x = max(0, min(xs)), min(w_img, max(xs))
        min_y, max_y = max(0, min(ys)), min(h_img, max(ys))
        w = max_x - min_x
        h = max_y - min_y
        
        # 擴大框的範圍（避免只框到臉部特徵，不含頭髮、下巴等）
        expand_top = int(h * 0.3)    # 向上擴展 30%（包含額頭）
        expand_sides = int(w * 0.15)  # 左右各擴展 15%
        
        min_x = max(0, min_x - expand_sides)
        max_x = min(w_img, max_x + expand_sides)
        min_y = max(0, min_y - expand_top)
        max_y = min(h_img, max_y)
        
        # 重新計算擴展後的寬高
        w = max_x - min_x
        h = max_y - min_y
        
        if w > 0 and h > 0:
            boxes.append((min_x, min_y, w, h))
    
    # 步驟 4：使用 NMS（非極大值抑制）移除重疊的框
    keep_idx = _nms_indices(boxes)
    if not keep_idx:
        return [], np.array([])
    
    # 只保留沒有重疊的人臉
    filtered_landmarks = [mp_result.face_landmarks[i] for i in keep_idx]
    filtered_boxes = [boxes[i] for i in keep_idx]
    
    return filtered_landmarks, np.array(filtered_boxes)


def detect_faces_bgr(image_bgr: np.ndarray):
    """
    簡化版的人臉偵測（只回傳邊界框）
    
    參數：
        image_bgr: OpenCV 圖片（BGR 格式）
    
    回傳：
        人臉邊界框陣列 [(x, y, w, h), ...]
    """
    if FACE_LANDMARKER_IMAGE is None:
        return np.array([])
    _, boxes = _detect_landmarks_bgr(image_bgr, FACE_LANDMARKER_IMAGE, timestamp_ms=None)
    return boxes


def _select_primary_face(faces):
    """
    從多張人臉中選出最大的那張（主要人臉）
    
    參數：
        faces: 人臉邊界框陣列
    
    回傳：
        只包含最大人臉的陣列
    """
    if faces is None or len(faces) == 0:
        return faces
    largest = max(faces, key=lambda f: f[2] * f[3])  # 依面積排序
    return np.array([largest])


def _sort_faces(faces):
    """
    依照位置排序人臉（從左到右，從上到下）
    用於影片處理時追蹤同一張人臉
    
    參數：
        faces: 人臉邊界框陣列
    
    回傳：
        排序後的人臉陣列
    """
    if faces is None or len(faces) == 0:
        return faces
    return np.array(sorted(faces, key=lambda f: (f[0], f[1])))


def _smooth_faces(prev_faces, curr_faces, alpha=0.7):
    """
    平滑人臉框的位置（影片處理專用）
    避免人臉框在影片中抖動，讓馬賽克或遮罩更穩定
    
    參數：
        prev_faces: 上一幀的人臉框
        curr_faces: 目前這一幀的人臉框
        alpha: 平滑係數（0-1），預設 0.7
    
    回傳：
        平滑後的人臉框陣列
    
    原理：使用「加權平均」混合上一幀和目前這一幀的位置
    - 移動快時：跟隨度高（避免延遲）
    - 移動慢時：平滑度高（避免抖動）
    """
    # 如果沒有目前幀的人臉，使用上一幀
    if curr_faces is None or len(curr_faces) == 0:
        return prev_faces
    
    # 如果沒有上一幀，直接使用目前幀
    if prev_faces is None or len(prev_faces) == 0:
        return curr_faces
    
    # 排序人臉（確保對應到同一張臉）
    prev_faces = _sort_faces(prev_faces)
    curr_faces = _sort_faces(curr_faces)
    
    # 如果人臉數量不同，無法對應，直接使用目前幀
    if len(prev_faces) != len(curr_faces):
        return curr_faces
    
    # 計算平滑後的位置
    smoothed = []
    for (px, py, pw, ph), (cx, cy, cw, ch) in zip(prev_faces, curr_faces):
        # 計算人臉的移動距離（相對於臉的大小）
        face_size = max(pw, ph)
        if face_size == 0:
            face_size = 1
        
        dx = abs(cx - px) / face_size  # x 方向的相對移動
        dy = abs(cy - py) / face_size  # y 方向的相對移動
        movement = (dx**2 + dy**2) ** 0.5  # 總移動距離
        
        # 根據移動速度調整平滑係數
        if movement > 0.3:  # 快速移動
            adaptive_alpha = 0.2  # 降低平滑（快速跟隨）
        elif movement > 0.1:  # 中速移動
            adaptive_alpha = 0.4
        else:  # 慢速移動
            adaptive_alpha = 0.7  # 提高平滑（減少抖動）
        
        # 計算平滑後的座標（加權平均）
        sx = int(round(adaptive_alpha * px + (1 - adaptive_alpha) * cx))
        sy = int(round(adaptive_alpha * py + (1 - adaptive_alpha) * cy))
        sw = int(round(adaptive_alpha * pw + (1 - adaptive_alpha) * cw))
        sh = int(round(adaptive_alpha * ph + (1 - adaptive_alpha) * ch))
        smoothed.append((sx, sy, sw, sh))
    
    return np.array(smoothed)


def _compute_iou(a, b):
    """
    計算兩個矩形的 IoU（Intersection over Union，重疊度）
    
    參數：
        a, b: 兩個矩形 (x, y, w, h)
    
    回傳：
        0.0-1.0 之間的數值，0 表示完全不重疊，1 表示完全重疊
    
    用途：NMS（非極大值抑制）時判斷兩個人臉框是否重疊
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah  # a 的右下角
    bx2, by2 = bx + bw, by + bh  # b 的右下角
    
    # 計算交集矩形
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # 交集面積
    
    if inter_area == 0:
        return 0.0
    
    # 計算聯集面積
    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    
    return inter_area / float(union_area)


def _nms_indices(faces, iou_thresh=0.45):
    """
    NMS（非極大值抑制）- 移除重疊的人臉框
    
    參數：
        faces: 人臉框陣列 [(x, y, w, h), ...]
        iou_thresh: IoU 閾值，超過此值視為重疊
    
    回傳：
        要保留的人臉索引列表
    
    原理：
    1. 依面積大小排序（大的優先）
    2. 保留最大的，移除與它重疊的
    3. 重複直到所有人臉都處理完
    """
    if faces is None or len(faces) == 0:
        return []
    
    faces = np.array(faces)
    scores = faces[:, 2] * faces[:, 3]  # 用面積當作分數
    order = scores.argsort()[::-1]  # 從大到小排序
    keep_idx = []
    
    while order.size > 0:
        i = order[0]
        keep_idx.append(int(i))  # 保留最大的
        
        if order.size == 1:
            break
        
        # 檢查剩下的人臉是否與目前這張重疊
        rest = []
        for j in order[1:]:
            if _compute_iou(faces[i], faces[j]) <= iou_thresh:
                rest.append(j)  # 不重疊，保留
        
        order = np.array(rest, dtype=int)
    
    return keep_idx


# ==================== 人臉資料儲存與載入 ====================

def _save_faces_metadata(image_bgr: np.ndarray, faces, media_id: str):
    """
    儲存偵測到的人臉資訊
    1. 裁切每張人臉並儲存為獨立圖片
    2. 將人臉位置資訊儲存為 JSON
    
    參數：
        image_bgr: 原始圖片
        faces: 人臉框陣列
        media_id: 媒體檔案 ID
    
    回傳：
        人臉資訊列表 [{"id": 0, "x": 100, "y": 200, "w": 80, "h": 100, "file": "xxx_face_0.jpg"}, ...]
    """
    items = []
    if faces is None or len(faces) == 0:
        faces = []
    
    for idx, (x, y, w, h) in enumerate(faces):
        # 裁切人臉區域
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = max(0, x + w), max(0, y + h)
        crop = image_bgr[y1:y2, x1:x2]
        
        # 儲存人臉圖片
        crop_name = f"{media_id}_face_{idx}.jpg"
        crop_path = PREVIEW_DIR / crop_name
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        
        # 記錄人臉資訊
        items.append({
            "id": idx,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "file": crop_name
        })
    
    # 將資訊儲存為 JSON 檔案
    meta_path = METADATA_DIR / f"{media_id}_faces.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f)
    
    return items


def _load_faces_metadata(media_id: str):
    """
    載入之前儲存的人臉資訊
    
    參數：
        media_id: 媒體檔案 ID
    
    回傳：
        人臉資訊列表，如果找不到則回傳空列表
    """
    meta_path = METADATA_DIR / f"{media_id}_faces.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_faces_by_indices(faces, selected_ids):
    """
    根據使用者選擇，篩選要處理的人臉
    
    參數：
        faces: 所有人臉框
        selected_ids: 使用者選擇的人臉 ID 列表
    
    回傳：
        篩選後的人臉框陣列
    """
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
    """
    根據使用者選擇，篩選要處理的人臉特徵點
    
    參數：
        landmarks: 所有人臉特徵點
        selected_ids: 使用者選擇的人臉 ID 列表
    
    回傳：
        篩選後的特徵點列表
    """
    if not landmarks:
        return []
    if not selected_ids:
        return landmarks
    
    selected = []
    for idx in selected_ids:
        if 0 <= idx < len(landmarks):
            selected.append(landmarks[idx])
    
    return selected


# ==================== 影像處理函式 ====================

def draw_face_boxes(image_bgr: np.ndarray, faces):
    """
    在圖片上繪製人臉框（預覽用）
    
    參數：
        image_bgr: 原始圖片
        faces: 人臉框陣列
    
    回傳：
        繪製了黃色矩形框的圖片
    """
    boxed = image_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 黃色框，線寬 2
    return boxed


def apply_mosaic(image_bgr: np.ndarray, faces, block_size=12):
    """
    對人臉區域套用馬賽克效果
    
    參數：
        image_bgr: 原始圖片
        faces: 人臉框陣列
        block_size: 馬賽克方塊大小（預設 12 像素）
    
    回傳：
        處理後的圖片
    
    原理：
    1. 將人臉區域縮小（除以 block_size）
    2. 再放大回原尺寸（使用最近鄰插值）
    3. 產生「像素化」的馬賽克效果
    """
    result = image_bgr.copy()
    
    for (x, y, w, h) in faces:
        # 取得人臉區域（ROI: Region of Interest）
        roi = result[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        
        # 縮小圖片（製造馬賽克效果）
        small_w = max(1, w // block_size)
        small_h = max(1, h // block_size)
        small = cv2.resize(roi, (small_w, small_h))
        
        # 放大回原尺寸（使用最近鄰插值，保持方塊狀）
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 替換原區域
        result[y : y + h, x : x + w] = mosaic
    
    return result


def _adaptive_alpha(prev_boxes, curr_boxes, base=0.5, min_alpha=0.2):
    """
    根據眼部區域的移動速度動態調整平滑係數
    
    參數：
        prev_boxes: 上一幀的眼部區域
        curr_boxes: 目前這一幀的眼部區域
        base: 基礎平滑係數
        min_alpha: 最小平滑係數
    
    回傳：
        調整後的平滑係數
    
    原理：
    - 快速移動：降低平滑（快速跟隨，避免延遲）
    - 慢速移動：提高平滑（減少抖動）
    """
    if not prev_boxes or not curr_boxes:
        return base
    
    prev_boxes = sorted(prev_boxes, key=lambda b: (b[0], b[1]))
    curr_boxes = sorted(curr_boxes, key=lambda b: (b[0], b[1]))
    
    if len(prev_boxes) != len(curr_boxes):
        return base
    
    max_relative_move = 0
    
    for (px1, py1, px2, py2), (cx1, cy1, cx2, cy2) in zip(prev_boxes, curr_boxes):
        # 計算眼部區域大小
        box_width = max(px2 - px1, cx2 - cx1, 1)
        box_height = max(py2 - py1, cy2 - cy1, 1)
        box_size = (box_width + box_height) / 2.0
        
        # 計算中心點的移動距離
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        ccx = (cx1 + cx2) / 2.0
        ccy = (cy1 + cy2) / 2.0
        move_dist = ((ccx - pcx)**2 + (ccy - pcy)**2) ** 0.5
        
        # 計算相對移動距離（相對於區域大小）
        relative_move = move_dist / box_size
        max_relative_move = max(max_relative_move, relative_move)
    
    # 根據移動速度選擇平滑係數
    if max_relative_move > 0.4:  # 快速移動
        return min_alpha
    if max_relative_move > 0.15:  # 中速移動
        return 0.3
    return base  # 慢速移動


def _smooth_boxes(prev_boxes, curr_boxes, alpha=0.5):
    """
    使用加權平均減少抖動
    
    參數：
        prev_boxes: 上一幀的眼部區域 [(x1, y1, x2, y2), ...]
        curr_boxes: 目前這一幀的眼部區域
        alpha: 平滑係數（0-1）
    
    回傳：
        平滑後的區域列表
    """
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
        # 使用加權平均計算新位置
        sx1 = int(round(alpha * px1 + (1 - alpha) * cx1))
        sy1 = int(round(alpha * py1 + (1 - alpha) * cy1))
        sx2 = int(round(alpha * px2 + (1 - alpha) * cx2))
        sy2 = int(round(alpha * py2 + (1 - alpha) * cy2))
        smoothed.append((sx1, sy1, sx2, sy2))
    
    return smoothed


def _get_points(landmarks, idxs, w_img, h_img):
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
        band_w = int(eye_dist * 2.4) 
        band_h = int(eye_dist * 0.75) 
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
    """
    載入替換用的圖片（支援透明度）
    
    參數：
        path: 圖片檔案路徑
    
    回傳：
        包含 Alpha 通道的圖片（RGBA 格式）
    """
    overlay = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return None
    
    # 如果圖片沒有 Alpha 通道，自動加上（完全不透明）
    if overlay.shape[2] == 3:
        alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=np.uint8) * 255
        overlay = np.concatenate([overlay, alpha], axis=2)
    
    return overlay


def apply_face_replace(image_bgr: np.ndarray, faces, overlay_rgba: np.ndarray):
    """
    用自訂圖片替換人臉
    
    參數：
        image_bgr: 原始圖片
        faces: 人臉框陣列
        overlay_rgba: 要替換的圖片（RGBA 格式）
    
    回傳：
        處理後的圖片
    
    原理：使用 Alpha 混合，支援半透明效果
    """
    result = image_bgr.copy()
    
    for (x, y, w, h) in faces:
        if w <= 0 or h <= 0:
            continue
        
        # 將替換圖片調整為與人臉相同大小
        overlay_resized = cv2.resize(overlay_rgba, (w, h))
        overlay_rgb = overlay_resized[:, :, :3]  # RGB 部分
        alpha = overlay_resized[:, :, 3:4] / 255.0  # Alpha 通道（0-1）
        
        # 取得原圖的人臉區域
        roi = result[y : y + h, x : x + w]
        if roi.shape[:2] != overlay_rgb.shape[:2]:
            continue
        
        # Alpha 混合：新圖 * alpha + 原圖 * (1 - alpha)
        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        result[y : y + h, x : x + w] = blended
    
    return result


def _open_video_writer(out_base: Path, fps: float, size: tuple[int, int]):
    """
    建立影片寫入器（嘗試多種編碼格式）
    
    參數：
        out_base: 輸出檔案路徑（不含副檔名）
        fps: 影格率
        size: 影片尺寸 (寬, 高)
    
    回傳：
        (writer, output_path) 或 (None, None)
    """
    # 依序嘗試不同的編碼器
    candidates = [("avc1", ".mp4"), ("mp4v", ".mp4"), ("VP80", ".webm")]
    
    for fourcc_name, ext in candidates:
        out_path = out_base.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        
        if writer.isOpened():
            return writer, out_path
    
    return None, None


def _save_preview(image_bgr: np.ndarray, name: str):
    """
    儲存預覽圖片
    
    參數：
        image_bgr: 圖片
        name: 檔案名稱（不含副檔名）
    
    回傳：
        儲存的檔案路徑
    """
    preview_path = PREVIEW_DIR / f"{name}.jpg"
    cv2.imwrite(str(preview_path), image_bgr)
    return preview_path



@app.route("/set_language/<lang>")
def set_language(lang):
    """
    切換語言
    將選擇的語言儲存到 session
    """
    if lang in app.config['LANGUAGES']:
        session['lang'] = lang
    
    # 取得來源頁面
    referrer = request.referrer or url_for('index')
    
    # 如果來源是只接受 POST 的路由（如 /upload），則導向首頁
    # 避免用 GET 方法訪問這些路由導致錯誤
    post_only_routes = ['/upload']
    for route in post_only_routes:
        if route in referrer:
            return redirect(url_for('index'))
    
    # 其他情況導回上一頁
    return redirect(referrer)


@app.route("/")
def index():
    """
    首頁 - 上傳照片/影片的頁面
    需要登入才能使用
    """
    if not current_user.is_authenticated:
        return redirect(url_for("auth.login"))
    return render_template("index.html")


@app.route("/options/<media_id>")
@login_required
def options(media_id):
    """
    顯示處理選項頁面
    從資料庫讀取已上傳的媒體資料
    """
    # 從資料庫查詢媒體記錄
    media = Media.query.filter_by(media_id=media_id, user_id=current_user.id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    # 讀取人臉資料
    faces_json_path = METADATA_DIR / f"{media_id}_faces.json"
    faces_info = []
    if faces_json_path.exists():
        with open(faces_json_path, "r", encoding="utf-8") as f:
            faces_info = json.load(f)
    
    # 取得預覽圖片
    preview_files = list(PREVIEW_DIR.glob(f"{media_id}_preview.*"))
    preview_url = url_for("previews", filename=preview_files[0].name) if preview_files else ""
    
    is_video = media.file_type == "video"
    
    return render_template(
        "options.html",
        media_id=media_id,
        is_video=is_video,
        preview_url=preview_url,
        faces=faces_info,
    )


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """
    上傳照片/影片並偵測人臉
    
    流程：
    1. 接收使用者上傳的檔案
    2. 偵測人臉並標記位置
    3. 儲存到資料庫
    4. 重定向到 options 頁面
    """
    # 步驟 1：取得上傳的檔案
    file = request.files.get("media")
    if not file or not file.filename:
        abort(400, "未提供檔案")
    
    # 步驟 2：取得人臉偵測靈敏度（0.3-0.9）
    try:
        sensitivity = float(request.form.get("sensitivity", 0.6))
        sensitivity = max(0.3, min(0.9, sensitivity))  # 限制在合理範圍內
    except:
        sensitivity = 0.6  # 預設值
    
    # 步驟 3：檢查檔案格式
    original_filename = file.filename
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
        abort(400, "檔案格式不支援")

    # 步驟 4：產生唯一的檔案 ID 並儲存檔案
    media_id = _generate_media_id()
    if ext in ALLOWED_IMAGE_EXT:
        saved_path = UPLOAD_IMAGE_DIR / f"{media_id}{ext}"
        file_type = "image"
    else:
        saved_path = UPLOAD_VIDEO_DIR / f"{media_id}{ext}"
        file_type = "video"
    file.save(saved_path)

    if _is_image(saved_path):
        image = cv2.imread(str(saved_path))
        face_detector = _create_face_landmarker_image(sensitivity)
        _, faces = _detect_landmarks_bgr(image, face_detector, None)
        faces_info = _save_faces_metadata(image, faces, media_id)
        preview = draw_face_boxes(image, faces)
        preview_path = _save_preview(preview, f"{media_id}_preview")
        
        media_record = Media(
            media_id=media_id,
            original_filename=original_filename,
            file_type=file_type,
            upload_path=str(saved_path),
            face_count=len(faces_info),
            status="uploaded",
            user_id=current_user.id,  
        )
        db.session.add(media_record)
        db.session.commit()
        
        # 重定向到 options 頁面
        return redirect(url_for("options", media_id=media_id))

    cap = cv2.VideoCapture(str(saved_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        abort(400, "無法讀取影片")
    # 使用自訂靈敏度創建偵測器
    face_detector = _create_face_landmarker_image(sensitivity)
    _, faces = _detect_landmarks_bgr(frame, face_detector, None)
    faces_info = _save_faces_metadata(frame, faces, media_id)
    preview = draw_face_boxes(frame, faces)
    preview_path = _save_preview(preview, f"{media_id}_preview")
    
    media_record = Media(
        media_id=media_id,
        original_filename=original_filename,
        file_type=file_type,
        upload_path=str(saved_path),
        face_count=len(faces_info),
        status="uploaded",
        user_id=current_user.id, 
    )
    db.session.add(media_record)
    db.session.commit()
    
    # 重定向到 options 頁面
    return redirect(url_for("options", media_id=media_id))


@app.route("/result/<media_id>")
@login_required
def result(media_id):
    """
    顯示處理結果頁面
    從資料庫讀取已處理的媒體資料
    """
    # 從資料庫查詢媒體記錄
    media = Media.query.filter_by(media_id=media_id, user_id=current_user.id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    if media.status != "processed" or not media.output_path:
        abort(400, "檔案尚未處理完成")
    
    # 取得結果檔案 URL
    is_video = media.file_type == "video"
    output_filename = Path(media.output_path).name
    
    if is_video:
        result_url = url_for("output_videos", filename=output_filename)
    else:
        result_url = url_for("output_images", filename=output_filename)
    
    return render_template(
        "result.html",
        is_video=is_video,
        result_url=result_url,
    )


@app.route("/process", methods=["POST"])
@login_required
def process():
    """
    處理照片/影片
    依照使用者選擇的模式（mosaic/eyes/replace）進行處理
    
    流程：
    1. 取得處理參數（media_id, mode, 選擇的人臉）
    2. 載入原始檔案
    3. 根據模式套用效果（馬賽克/遮眼/替換）
    4. 儲存處理後的檔案
    5. 重定向到結果頁面
    """
    # 步驟 1：取得處理參數
    media_id = request.form.get("media_id", "").strip()
    mode = request.form.get("mode", "").strip()  # mosaic / eyes / replace
    
    # 取得使用者選擇的人臉 ID
    face_ids_raw = request.form.getlist("face_ids")
    selected_ids = []
    for item in face_ids_raw:
        try:
            selected_ids.append(int(item))
        except ValueError:
            continue
    
    if not media_id:
        abort(400, "缺少 media_id")

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
        
        media_record = Media.query.filter_by(media_id=media_id).first()
        if media_record:
            media_record.output_path = str(out_path)
            media_record.process_mode = mode
            media_record.status = "processed"
            media_record.processed_at = datetime.now()
            db.session.commit()
        
        # 重定向到結果頁面
        return redirect(url_for("result", media_id=media_id))

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
    
    media_record = Media.query.filter_by(media_id=media_id).first()
    if media_record:
        media_record.output_path = str(out_path)
        media_record.process_mode = mode
        media_record.status = "processed"
        media_record.processed_at = datetime.now()
        db.session.commit()
    
    # 重定向到結果頁面
    return redirect(url_for("result", media_id=media_id))


@app.route("/outputs/images/<path:filename>")
@login_required
def output_images(filename):
    """
    提供處理後的照片下載/顯示
    需要登入才能存取
    """
    return send_from_directory(OUTPUT_IMAGE_DIR, filename, as_attachment=False)


@app.route("/outputs/videos/<path:filename>")
@login_required
def output_videos(filename):
    """
    提供處理後的影片下載/播放
    需要登入才能存取
    """
    return send_from_directory(OUTPUT_VIDEO_DIR, filename, as_attachment=False)


@app.route("/previews/<path:filename>")
@login_required
def previews(filename):
    """
    提供預覽圖片（含人臉框）
    需要登入才能存取
    """
    return send_from_directory(PREVIEW_DIR, filename, as_attachment=False)


# ==================== 啟動應用程式 ====================

if __name__ == "__main__":
    # 開發模式：在本機 5000 port 啟動，並開啟除錯模式
    app.run(host="0.0.0.0", port=5000, debug=True)
