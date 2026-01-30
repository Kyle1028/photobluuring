"""
主程式檔案：處理照片/影片上傳、人臉偵測、隱私處理
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2  # OpenCV：影像處理
import numpy as np  # NumPy：數值運算
from flask import Flask, render_template, request, send_from_directory, abort, url_for, redirect, session, flash, jsonify
from flask_login import login_required, current_user
from flask_babel import Babel, gettext as _, lazy_gettext

from sqlalchemy import or_, and_
from core.auth import init_auth  # 認證系統
from core.models import (
    db,
    Media,
    Exhibition,
    ExhibitionPhoto,
    User,
    ExhibitionFloor,
    ExhibitionCell,
    ExhibitionMergedRegion,
    _media_id_from_seq,
    _refresh_media_id_suffix,
)  # 資料庫模型
from core.media_processor import MediaProcessor  # 媒體處理模組

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
EXHIBITION_DIR = BASE_DIR / "exhibitions"              # 展覽照片目錄

# 自動建立所有需要的目錄（如果不存在）
for d in (UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, PREVIEW_DIR, METADATA_DIR, EXHIBITION_DIR):
    d.mkdir(parents=True, exist_ok=True)


def generate_cells_for_floor(floor: ExhibitionFloor) -> list[ExhibitionCell]:
    """
    根據樓層的實際尺寸與 grid_size，自動產生該樓層的區域 Cell。
    - 每個樓層的 Cell 代碼從 C000001 重新編號。
    - 依「上到下、左到右」順序產生 row/col 與 cell_code。
    """
    if not floor.width_meters or not floor.height_meters or not floor.grid_size:
        return []

    cols = int(floor.width_meters / floor.grid_size)
    rows = int(floor.height_meters / floor.grid_size)

    cells: list[ExhibitionCell] = []
    idx = 1

    for row in range(rows):
        for col in range(cols):
            code = f"C{idx:06d}"
            cell = ExhibitionCell(
                floor_id=floor.id,
                cell_code=code,
                row=row,
                col=col,
                name=f"區域 {code}",
                is_active=True,
            )
            cells.append(cell)
            idx += 1

    if cells:
        db.session.add_all(cells)

    return cells


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

# 初始化管理員系統（延遲導入避免循環依賴）
def init_admin_system():
    from core.admin import init_admin
    init_admin(app)

init_admin_system()

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
        lang = session['lang']
        # 轉換為 Babel 可識別的格式
        if lang == 'en':
            return 'en'
        elif lang == 'zh_Hant_TW' or lang == 'zh':
            return 'zh_Hant_TW'
        return lang
    
    # 回傳預設語言（繁體中文）
    return app.config['BABEL_DEFAULT_LOCALE']


# 設定 Babel 的語言選擇函式
babel.init_app(app, locale_selector=get_locale)

# 讓模板可以訪問 get_locale 函數和當前語言
@app.context_processor
def inject_get_locale():
    def get_current_locale():
        """獲取當前語言代碼（簡化版，用於模板判斷）"""
        if 'lang' in session:
            return session['lang']
        return 'zh_Hant_TW'
    return dict(get_locale=get_current_locale, current_locale=get_current_locale())

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




def _temp_media_id() -> str:
    """上傳時先用的暫時 ID，存檔與建立 Media 後再以 _apply_real_media_id 換成 8+15+4 格式。"""
    return "tmp" + uuid.uuid4().hex[:14]


def _rename_media_ids(old_id: str, new_id: str, media_record: Media):
    """將所有以 old_id 命名的檔案改為 new_id，並更新 media_record 與 ExhibitionPhoto 的路徑。"""
    # 1. 上傳檔（uploads/images 或 videos 底下的 {old_id}.ext）
    for base in (UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR):
        for p in base.rglob(f"{old_id}.*"):
            ext = p.suffix
            new_p = p.parent / f"{new_id}{ext}"
            if p != new_p:
                p.rename(new_p)
                if media_record and media_record.upload_path == str(p):
                    media_record.upload_path = str(new_p)
    # 2. 預覽圖
    for p in PREVIEW_DIR.glob(f"{old_id}_preview.*"):
        new_p = p.parent / f"{new_id}_preview{p.suffix}"
        p.rename(new_p)
    # 2b. 預覽目錄下的人臉截圖（{media_id}_face_0.jpg 等）
    for p in PREVIEW_DIR.glob(f"{old_id}_face_*.jpg"):
        rest = p.name[len(old_id) :]
        p.rename(p.parent / f"{new_id}{rest}")
    # 3. metadata（faces.json、face crops）
    j = METADATA_DIR / f"{old_id}_faces.json"
    if j.exists():
        j.rename(METADATA_DIR / f"{new_id}_faces.json")
    for p in METADATA_DIR.glob(f"{old_id}_face_*.jpg"):
        rest = p.name[len(old_id) :]
        p.rename(METADATA_DIR / f"{new_id}{rest}")
    # 4. 輸出品（outputs 底下的 {old_id}_out.*）
    for base in (OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR):
        for p in base.rglob(f"{old_id}_out.*"):
            new_p = p.parent / f"{new_id}_out{p.suffix}"
            p.rename(new_p)
            if media_record and getattr(media_record, "output_path", None) == str(p):
                media_record.output_path = str(new_p)
    # 5. overlay（上傳目錄下的 {old_id}_overlay.*）
    for base in (UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR):
        for p in base.rglob(f"{old_id}_overlay.*"):
            new_p = p.parent / f"{new_id}_overlay{p.suffix}"
            p.rename(new_p)
    # 6. 更新 media 紀錄
    if media_record:
        media_record.media_id = new_id
        if media_record.upload_path and old_id in media_record.upload_path:
            media_record.upload_path = media_record.upload_path.replace(old_id, new_id)
        if getattr(media_record, "output_path", None) and old_id in (media_record.output_path or ""):
            media_record.output_path = (media_record.output_path or "").replace(old_id, new_id)
    # 7. 更新展覽照片路徑（相對路徑字串中的 old_id）
    if media_record and getattr(media_record, "exhibition_id", None):
        for ep in ExhibitionPhoto.query.filter_by(exhibition_id=media_record.exhibition_id).all():
            if ep.photo_path and old_id in ep.photo_path:
                ep.photo_path = ep.photo_path.replace(old_id, new_id)
            if ep.thumbnail_path and old_id in ep.thumbnail_path:
                ep.thumbnail_path = ep.thumbnail_path.replace(old_id, new_id)


def _apply_real_media_id(media_record: Media) -> str:
    """將暫時 media_id 換成正式格式 8+15位序號+4碼隨機，並 rename 相關檔案。回傳新 media_id。"""
    old_id = media_record.media_id
    new_id = _media_id_from_seq(media_record.id)
    if old_id == new_id:
        return new_id
    _rename_media_ids(old_id, new_id, media_record)
    return new_id


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
    首頁 - 展覽列表頁面
    顯示所有公開的展覽，讓使用者選擇要查看哪個展覽
    支援 ?q=關鍵字 搜尋展覽標題與描述
    """
    q = request.args.get("q", "").strip()
    base_query = Exhibition.query.filter_by(is_published=True)
    if q:
        pattern = f"%{q}%"
        base_query = base_query.filter(
            or_(
                Exhibition.title.ilike(pattern),
                and_(Exhibition.description.isnot(None), Exhibition.description.ilike(pattern)),
            )
        )
    exhibitions = base_query.order_by(Exhibition.created_at.desc()).all()
    
    # 檢查並清理封面圖片不存在的展覽
    for exhibition in exhibitions:
        if exhibition.cover_image:
            # 處理 Windows 路徑分隔符
            cover_image_path = exhibition.cover_image.replace('\\', '/')
            cover_path = Path(cover_image_path)
            if cover_path.is_absolute():
                full_path = cover_path
            else:
                full_path = BASE_DIR / cover_path
            
            # 如果封面圖片檔案不存在，清除資料庫中的路徑
            if not full_path.exists():
                try:
                    exhibition.cover_image = None
                    db.session.commit()
                except Exception:
                    db.session.rollback()
    
    return render_template("exhibitions_list.html", exhibitions=exhibitions, search_query=q)


@app.route("/exhibition/<exhibition_public_id>")
def exhibition_detail(exhibition_public_id):
    """
    展覽詳情頁面（對外使用 public_id，不可猜）
    顯示展覽的詳細資訊和所有照片
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 檢查展覽是否公開
    if not exhibition.is_published:
        # 只有展覽創建者或管理員可以查看未公開的展覽
        if not current_user.is_authenticated or not current_user.can_manage_exhibition(exhibition):
            abort(403, "此展覽尚未公開")
    
    # 取得展覽的所有照片，按顯示順序排列
    all_photos = ExhibitionPhoto.query.filter_by(exhibition_id=exhibition.id).order_by(ExhibitionPhoto.display_order).all()
    
    # 過濾掉檔案不存在的照片
    photos = []
    for photo in all_photos:
        photo_path = Path(photo.photo_path)
        if photo_path.is_absolute():
            full_path = photo_path
        else:
            full_path = BASE_DIR / photo_path
        
        # 如果檔案存在，才加入列表
        if full_path.exists():
            # 預設：區域資訊為空列表，避免模板存取錯誤
            photo.cell_codes = []
            
            # 嘗試找出對應的 Media 記錄，以便顯示此照片所屬的展覽區域（Cell）
            try:
                photo_filename = full_path.name
                linked_media = Media.query.filter(
                    (Media.upload_path.like(f"%{photo_filename}")) |
                    (Media.output_path.like(f"%{photo_filename}")) |
                    (Media.upload_path == str(photo.photo_path)) |
                    (Media.upload_path == str(full_path)) |
                    (Media.output_path == str(photo.photo_path)) |
                    (Media.output_path == str(full_path))
                ).first()
                
                # 若找到對應的 Media，整理其所有關聯的區域代碼（依樓層與 cell_code 排序）
                if linked_media and linked_media.cells:
                    labels = []
                    for cell in linked_media.cells:
                        try:
                            floor_code = getattr(cell.floor, "floor_code", "")
                        except Exception:
                            floor_code = ""
                        # 優先顯示 F+C 組合，若無樓層代碼則只顯示 C 碼
                        label = f"{floor_code} {cell.cell_code}".strip()
                        labels.append(label)
                    # 去重並排序，避免重複顯示
                    unique_labels = sorted(set(labels))
                    photo.cell_codes = unique_labels
            except Exception:
                # 若關聯查詢失敗，不影響基礎照片顯示，只是不顯示區域資訊
                photo.cell_codes = []
            
            photos.append(photo)
        else:
            # 檔案不存在，自動刪除資料庫記錄
            try:
                db.session.delete(photo)
            except Exception:
                pass  # 如果刪除失敗，忽略錯誤
    
    # 提交刪除操作
    if len(photos) < len(all_photos):
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
    
    # 載入樓層資訊（如果有）
    floors = list(exhibition.floors)
    floors.sort(key=lambda f: f.floor_code)
    
    # 預設顯示第一個樓層（或由參數指定）
    selected_floor_code = request.args.get("floor", floors[0].floor_code if floors else None)
    selected_floor = None
    cells = []
    
    cell_groups = []  # 依合併區分組，展覽頁依區塊顯示
    merged_regions_for_plan = []  # 供平面圖繪製合併區名稱（僅含合併區）
    if floors and selected_floor_code:
        selected_floor = next((f for f in floors if f.floor_code == selected_floor_code), floors[0])
        if selected_floor:
            # 只獲取有效的網格（is_active=True）
            _cells = [c for c in selected_floor.cells if c.is_active]
            _cells.sort(key=lambda c: (c.row, c.col))
            # 轉成可 JSON 序列化的 dict（供 template |tojson 使用）
            def _cell_dict(c):
                return {
                    "id": c.id,
                    "cell_code": c.cell_code,
                    "row": c.row,
                    "col": c.col,
                    "name": c.name,
                    "is_active": bool(c.is_active),
                    "media_count": len(getattr(c, "media_files", []) or []),
                    "merged_region_id": getattr(c, "merged_region_id", None),
                }
            cells = [_cell_dict(c) for c in _cells]
            # 依合併區分組：先各合併區（依 display_order），再「未分區」
            merged_regions = list(getattr(selected_floor, "merged_regions", []) or [])
            merged_regions.sort(key=lambda r: (r.display_order, r.id))
            for mr in merged_regions:
                region_cells = [c for c in _cells if getattr(c, "merged_region_id", None) == mr.id]
                if region_cells:
                    region_cells.sort(key=lambda c: (c.row, c.col))
                    cell_groups.append({
                        "region_name": mr.name,
                        "region_id": mr.id,
                        "cells": [_cell_dict(c) for c in region_cells],
                    })
            unmerged = [c for c in _cells if not getattr(c, "merged_region_id", None)]
            if unmerged:
                unmerged.sort(key=lambda c: (c.row, c.col))
                cell_groups.append({
                    "region_name": None,
                    "region_id": None,
                    "cells": [_cell_dict(c) for c in unmerged],
                })
            # 供平面圖繪製合併區名稱（僅含合併區，每區 name + cells 的 row,col）
            merged_regions_for_plan = [
                {"name": g["region_name"], "cells": [{"row": c["row"], "col": c["col"]} for c in g["cells"]]}
                for g in cell_groups if g.get("region_name")
            ]
        else:
            merged_regions_for_plan = []
    
    return render_template(
        "exhibition_detail.html",
        exhibition=exhibition,
        photos=photos,
        floors=floors,
        selected_floor=selected_floor,
        cells=cells,
        cell_groups=cell_groups,
        merged_regions_for_plan=merged_regions_for_plan,
    )


@app.route("/exhibition/<exhibition_public_id>/cover")
def exhibition_cover(exhibition_public_id):
    """
    提供展覽封面圖片（對外使用 public_id）
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    if not exhibition.cover_image:
        abort(404, "此展覽沒有封面圖片")
    
    # 檢查展覽是否公開
    if not exhibition.is_published:
        # 只有展覽創建者或管理員可以查看未公開的展覽
        if not current_user.is_authenticated or not current_user.can_manage_exhibition(exhibition):
            abort(403, "此展覽尚未公開")
    
    # 構建完整路徑（處理 Windows 路徑分隔符）
    cover_image_path = exhibition.cover_image.replace('\\', '/')  # 統一使用正斜線
    cover_path = Path(cover_image_path)
    if cover_path.is_absolute():
        full_path = cover_path
    else:
        # 如果是相對路徑，從專案根目錄開始構建
        full_path = BASE_DIR / cover_path
    
    if not full_path.exists():
        abort(404, f"封面圖片不存在: {full_path}")
    
    return send_from_directory(full_path.parent, full_path.name, as_attachment=False)


@app.route("/exhibition/<exhibition_public_id>/floor/<floor_code>/image")
def exhibition_floor_image(exhibition_public_id, floor_code):
    """
    提供展覽樓層平面圖（對外使用 public_id）
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    floor = next((f for f in exhibition.floors if f.floor_code == floor_code), None)
    if not floor:
        abort(404, "找不到該樓層")
    
    # 檢查展覽是否公開
    if not exhibition.is_published:
        # 只有展覽創建者或管理員可以查看未公開的展覽
        if not current_user.is_authenticated or not current_user.can_manage_exhibition(exhibition):
            abort(403, "此展覽尚未公開")
    
    # 構建完整路徑
    image_path = Path(floor.image_path)
    if image_path.is_absolute():
        full_path = image_path
    else:
        full_path = BASE_DIR / image_path
    
    if not full_path.exists():
        abort(404, f"樓層平面圖不存在: {full_path}")
    
    return send_from_directory(full_path.parent, full_path.name, as_attachment=False)


@app.route("/exhibition/<exhibition_public_id>/floors/<floor_code>/cells/<cell_code>/media")
def get_cell_media(exhibition_public_id, floor_code, cell_code):
    """
    取得指定區域的所有媒體（JSON API，供 Modal 使用）
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 檢查展覽是否公開
    if not exhibition.is_published:
        if not current_user.is_authenticated or not current_user.can_manage_exhibition(exhibition):
            abort(403, "此展覽尚未公開")
    
    # 找到對應的樓層和區域
    floor = next((f for f in exhibition.floors if f.floor_code == floor_code), None)
    if not floor:
        abort(404, "找不到該樓層")
    
    cell = next((c for c in floor.cells if c.cell_code == cell_code), None)
    if not cell:
        abort(404, "找不到該區域")
    
    # 取得該區域的所有媒體（透過多對多關聯）
    media_list = []
    for media in cell.media_files:
        # 注意：/uploads/* 與 /outputs/* 目前需要登入才能存取。
        # 但展覽是給訪客看的，所以這裡改成回傳「展覽專用公開媒體 URL」。
        media_url = url_for("exhibition_media", exhibition_public_id=exhibition.public_id, media_id=media.media_id)
        preview_url = ""
        
        media_list.append({
            "media_id": media.media_id,
            "original_filename": media.original_filename,
            "file_type": media.file_type,
            "url": media_url,
            "preview_url": preview_url,
            "status": media.status,
        })
    
    merged_region_name = None
    if getattr(cell, "merged_region_id", None) and getattr(cell, "merged_region", None):
        merged_region_name = cell.merged_region.name
    return jsonify({
        "floor_code": floor_code,
        "cell_code": cell_code,
        "cell_name": cell.name or cell_code,
        "merged_region_name": merged_region_name,
        "media": media_list,
        "count": len(media_list),
    })


@app.route("/exhibition/<exhibition_public_id>/media/<media_id>")
def exhibition_media(exhibition_public_id, media_id):
    """
    提供展覽內媒體檔案（公開展覽可直接瀏覽；未公開需管理權限）
    - 圖片：回傳 processed(output) 優先，否則 upload
    - 影片：同上
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()

    # 檢查展覽是否公開
    if not exhibition.is_published:
        if not current_user.is_authenticated or not current_user.can_manage_exhibition(exhibition):
            abort(403, "此展覽尚未公開")

    media = Media.query.filter_by(media_id=media_id, exhibition_id=exhibition.id).first()
    if not media:
        abort(404, "找不到該媒體")

    file_path = None
    if media.status == "processed" and media.output_path:
        p = Path(media.output_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists():
            file_path = p

    if file_path is None and media.upload_path:
        p = Path(media.upload_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists():
            file_path = p

    if file_path is None:
        abort(404, "檔案不存在")

    return send_from_directory(file_path.parent, file_path.name, as_attachment=False)


@app.route("/exhibition/<exhibition_public_id>/photo/<int:photo_id>")
def exhibition_photo(exhibition_public_id, photo_id):
    """
    提供展覽照片/影片的存取（對外展覽用 public_id）
    支持圖片和影片
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 正常照片請求
    photo = ExhibitionPhoto.query.get_or_404(photo_id)
    
    # 檢查照片是否屬於該展覽
    if photo.exhibition_id != exhibition.id:
        abort(404, "照片不存在")
    
    # 檢查展覽是否公開
    if not exhibition.is_published:
        # 只有展覽創建者或管理員可以查看未公開的展覽
        if not current_user.is_authenticated or not current_user.can_manage_exhibition(exhibition):
            abort(403, "此展覽尚未公開")
    
    # 構建完整路徑
    photo_path = Path(photo.photo_path)
    if photo_path.is_absolute():
        full_path = photo_path
    else:
        # 如果是相對路徑，從專案根目錄開始構建
        full_path = BASE_DIR / photo_path
    
    if not full_path.exists():
        # 如果檔案不存在，返回錯誤
        abort(404, f"檔案不存在: {full_path}")
    
    return send_from_directory(full_path.parent, full_path.name, as_attachment=False)


@app.route("/options/<media_id>")
@login_required
def options(media_id):
    """
    顯示處理選項頁面（含人臉標註）
    從資料庫讀取已上傳的媒體資料。
    若尚無人臉資料（例如從展覽頁直接上傳的檔案），會先執行人臉偵測並寫入 metadata，
    再顯示人臉框與選項，避免「按處理卻沒有人臉」的情況。
    """
    media = Media.query.filter_by(media_id=media_id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    # 權限檢查：超級管理員、媒體上傳者、或媒體所屬展覽的創辦人可以處理
    has_permission = False
    if current_user.is_super_admin_role():
        has_permission = True
    elif media.user_id == current_user.id:
        has_permission = True
    elif media.exhibition_id:
        exhibition = db.session.get(Exhibition, media.exhibition_id)
        if exhibition and current_user.can_manage_exhibition(exhibition):
            has_permission = True
    
    if not has_permission:
        abort(403, "您沒有權限查看此檔案")
    
    faces_json_path = METADATA_DIR / f"{media_id}_faces.json"
    faces_info = []
    
    if faces_json_path.exists():
        with open(faces_json_path, "r", encoding="utf-8") as f:
            faces_info = json.load(f)
    elif media.upload_path:
        # 從展覽頁「直接上傳」的檔案沒有跑過人臉偵測，在此補做一次
        upload_path = Path(media.upload_path)
        if not upload_path.is_absolute():
            upload_path = BASE_DIR / upload_path
        if upload_path.exists():
            try:
                if media.file_type == "image":
                    img = cv2.imread(str(upload_path))
                    if img is not None:
                        fd = _create_face_landmarker_image(0.6)
                        _, faces = _detect_landmarks_bgr(img, fd, None)
                        faces_info = _save_faces_metadata(img, faces, media_id)
                        preview = draw_face_boxes(img, faces)
                        _save_preview(preview, f"{media_id}_preview")
                else:
                    cap = cv2.VideoCapture(str(upload_path))
                    ok, frame = cap.read()
                    cap.release()
                    if ok:
                        fd = _create_face_landmarker_image(0.6)
                        _, faces = _detect_landmarks_bgr(frame, fd, None)
                        faces_info = _save_faces_metadata(frame, faces, media_id)
                        preview = draw_face_boxes(frame, faces)
                        _save_preview(preview, f"{media_id}_preview")
                if faces_info and media:
                    media.face_count = len(faces_info)
                    db.session.commit()
            except Exception:
                pass  # 偵測失敗時仍顯示 options，faces_info 為空
    
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


@app.route("/upload/exhibition/<exhibition_public_id>/select-cells", methods=["GET", "POST"])
@login_required
def upload_exhibition_with_cells(exhibition_public_id):
    """
    上傳媒體並選擇區域（對外使用 public_id）
    GET：顯示平面圖選擇頁面
    POST：處理上傳並關聯選中的區域
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 檢查展覽是否有樓層
    floors = list(exhibition.floors)
    if not floors:
        flash(_("此展覽尚未設定樓層平面圖，請先編輯展覽並上傳平面圖"), "error")
        return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))
    
    if request.method == "POST":
        # 處理上傳
        file = request.files.get("media")
        if not file or not file.filename:
            flash(_("未提供檔案"), "error")
            return redirect(url_for("upload_exhibition_with_cells", exhibition_public_id=exhibition.public_id))
        
        # 取得選中的區域 cell_id 列表
        selected_cell_ids = request.form.getlist("selected_cells")
        if not selected_cell_ids:
            flash(_("請至少選擇一個區域"), "error")
            return redirect(url_for("upload_exhibition_with_cells", exhibition_public_id=exhibition.public_id))
        
        # 檢查檔案格式
        original_filename = file.filename
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
            flash(_("檔案格式不支援"), "error")
            return redirect(url_for("upload_exhibition_with_cells", exhibition_public_id=exhibition.public_id))
        
        # 暫時 ID，存檔後再換成 8+15位序號+4碼隨機
        media_id = _temp_media_id()
        
        # 按日期組織檔案並保存
        upload_date = datetime.now()
        if ext in ALLOWED_IMAGE_EXT:
            date_dir = UPLOAD_IMAGE_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
            date_dir.mkdir(parents=True, exist_ok=True)
            saved_path = date_dir / f"{media_id}{ext}"
            file_type = "image"
        else:
            date_dir = UPLOAD_VIDEO_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
            date_dir.mkdir(parents=True, exist_ok=True)
            saved_path = date_dir / f"{media_id}{ext}"
            file_type = "video"
        
        file.save(saved_path)
        
        # 將路徑轉換為相對路徑
        saved_path_obj = Path(saved_path)
        if saved_path_obj.is_absolute():
            try:
                relative_path = saved_path_obj.relative_to(BASE_DIR)
            except ValueError:
                relative_path = saved_path_obj
        else:
            relative_path = saved_path_obj
        
        # 生成預覽圖
        thumbnail_path = str(relative_path)
        if file_type == "video":
            try:
                cap = cv2.VideoCapture(str(saved_path))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    preview_path = PREVIEW_DIR / f"{media_id}_preview.jpg"
                    cv2.imwrite(str(preview_path), frame)
                    thumbnail_path = str(preview_path.relative_to(BASE_DIR))
            except Exception:
                pass
        else:
            try:
                image = cv2.imread(str(saved_path))
                if image is not None:
                    preview_path = PREVIEW_DIR / f"{media_id}_preview.jpg"
                    cv2.imwrite(str(preview_path), image)
                    thumbnail_path = str(preview_path.relative_to(BASE_DIR))
            except Exception:
                pass
        
        # 獲取該展覽現有的照片數量
        max_order = db.session.query(db.func.max(ExhibitionPhoto.display_order)).filter_by(
            exhibition_id=exhibition.id
        ).scalar() or -1
        
        # 創建 Media 記錄
        media_record = Media(
            media_id=media_id,
            original_filename=original_filename,
            file_type=file_type,
            upload_path=str(saved_path),
            face_count=0,
            status="uploaded",
            user_id=current_user.id,
            exhibition_id=exhibition.id,
        )
        db.session.add(media_record)
        
        # 創建展覽照片記錄
        exhibition_photo = ExhibitionPhoto(
            exhibition_id=exhibition.id,
            photo_path=str(relative_path),
            thumbnail_path=thumbnail_path,
            title=original_filename,
            description="",
            display_order=max_order + 1,
            created_at=datetime.now()
        )
        db.session.add(exhibition_photo)
        db.session.commit()
        _apply_real_media_id(media_record)
        db.session.commit()
        
        # 關聯選中的區域
        try:
            for cell_id_str in selected_cell_ids:
                cell_id = int(cell_id_str)
                cell = ExhibitionCell.query.get(cell_id)
                if cell and cell.floor.exhibition_id == exhibition.id:
                    # 使用多對多關聯
                    media_record.cells.append(cell)
            db.session.commit()
            flash(_("檔案已成功上傳並關聯到選中的區域"), "success")
        except Exception as e:
            db.session.rollback()
            flash(f"檔案上傳成功，但區域關聯失敗：{str(e)}", "warning")
        
        goto_media = request.args.get("redirect") == "media"
        if goto_media:
            return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition.public_id))
        return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))
    
    # GET：顯示選擇頁面
    # 預設顯示第一個樓層（或由參數指定）
    selected_floor_code = request.args.get("floor", floors[0].floor_code if floors else None)
    selected_floor = next((f for f in floors if f.floor_code == selected_floor_code), floors[0] if floors else None)
    
    if not selected_floor:
        flash(_("找不到指定的樓層"), "error")
        return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))
    
    # 載入該樓層的所有有效區域（只包含 is_active=True 的網格，依 row, col 排序）
    _cells = [c for c in selected_floor.cells if c.is_active]
    _cells.sort(key=lambda c: (c.row, c.col))
    cells = [
        {
            "id": c.id,
            "cell_code": c.cell_code,
            "row": c.row,
            "col": c.col,
            "name": c.name,
            "is_active": bool(c.is_active),
            "media_count": len(getattr(c, "media_files", []) or []),
            "merged_region_id": getattr(c, "merged_region_id", None),
        }
        for c in _cells
    ]
    # 合併區：供上傳時可一次選取整個合併區
    merged_regions = []
    merged_regions_for_plan = []
    for mr in sorted(getattr(selected_floor, "merged_regions", []) or [], key=lambda r: (r.display_order, r.id)):
        region_cells = [c for c in mr.cells if getattr(c, "is_active", True)]
        if region_cells:
            merged_regions.append({"id": mr.id, "name": mr.name, "cell_ids": [c.id for c in region_cells]})
            merged_regions_for_plan.append({
                "name": mr.name,
                "cells": [{"row": c.row, "col": c.col} for c in region_cells],
            })
    
    return render_template(
        "upload_grid_selection.html",
        exhibition=exhibition,
        floors=floors,
        selected_floor=selected_floor,
        cells=cells,
        merged_regions=merged_regions,
        merged_regions_for_plan=merged_regions_for_plan,
    )


@app.route("/upload/exhibition/<exhibition_public_id>", methods=["POST"])
@login_required
def upload_to_exhibition(exhibition_public_id):
    """
    直接上傳照片/影片到展覽（不進行隱私處理，對外使用 public_id）
    只保存檔案並添加到展覽照片列表中。
    若 URL 帶 ?redirect=media，成功後導回媒體管理該展覽列表。
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    goto_media = request.args.get("redirect") == "media"

    # 取得上傳的檔案
    file = request.files.get("media")
    if not file or not file.filename:
        flash(_("未提供檔案"), "error")
        if goto_media:
            return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition.public_id))
        return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))

    # 檢查檔案格式
    original_filename = file.filename
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
        flash(_("檔案格式不支援"), "error")
        if goto_media:
            return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition.public_id))
        return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))
    
    # 暫時 ID，存檔後再換成 8+15位序號+4碼隨機
    media_id = _temp_media_id()
    
    # 按日期組織檔案並保存
    upload_date = datetime.now()
    if ext in ALLOWED_IMAGE_EXT:
        date_dir = UPLOAD_IMAGE_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
        date_dir.mkdir(parents=True, exist_ok=True)
        saved_path = date_dir / f"{media_id}{ext}"
        file_type = "image"
    else:
        date_dir = UPLOAD_VIDEO_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
        date_dir.mkdir(parents=True, exist_ok=True)
        saved_path = date_dir / f"{media_id}{ext}"
        file_type = "video"
    
    file.save(saved_path)
    
    # 將路徑轉換為相對路徑（相對於 BASE_DIR）
    saved_path_obj = Path(saved_path)
    if saved_path_obj.is_absolute():
        try:
            relative_path = saved_path_obj.relative_to(BASE_DIR)
        except ValueError:
            relative_path = saved_path_obj
    else:
        relative_path = saved_path_obj
    
    # 生成預覽圖（圖片和影片都需要）
    thumbnail_path = str(relative_path)
    if file_type == "video":
        try:
            cap = cv2.VideoCapture(str(saved_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                # 保存第一幀作為預覽圖
                preview_path = PREVIEW_DIR / f"{media_id}_preview.jpg"
                cv2.imwrite(str(preview_path), frame)
                thumbnail_path = str(preview_path.relative_to(BASE_DIR))
        except Exception:
            pass  # 如果生成預覽圖失敗，使用原始路徑
    else:
        # 對於圖片，也生成預覽圖（直接複製圖片作為預覽圖）
        try:
            image = cv2.imread(str(saved_path))
            if image is not None:
                preview_path = PREVIEW_DIR / f"{media_id}_preview.jpg"
                cv2.imwrite(str(preview_path), image)
                thumbnail_path = str(preview_path.relative_to(BASE_DIR))
        except Exception:
            pass  # 如果生成預覽圖失敗，使用原始路徑
    
    # 獲取該展覽現有的照片數量，用於設定顯示順序
    max_order = db.session.query(db.func.max(ExhibitionPhoto.display_order)).filter_by(
        exhibition_id=exhibition.id
    ).scalar() or -1
    
    # 創建 Media 記錄（用於媒體管理）
    media_record = Media(
        media_id=media_id,
        original_filename=original_filename,
        file_type=file_type,
        upload_path=str(saved_path),
        face_count=0,  # 直接上傳不進行人臉偵測
        status="uploaded",  # 狀態為已上傳（未處理）
        user_id=current_user.id,
        exhibition_id=exhibition.id,
    )
    db.session.add(media_record)
    
    # 創建展覽照片記錄
    exhibition_photo = ExhibitionPhoto(
        exhibition_id=exhibition.id,
        photo_path=str(relative_path),
        thumbnail_path=thumbnail_path,
        title=original_filename,
        description="",
        display_order=max_order + 1,
        created_at=datetime.now()
    )
    db.session.add(exhibition_photo)
    db.session.commit()
    _apply_real_media_id(media_record)
    db.session.commit()
    
    flash(_("檔案已成功上傳到展覽"), "success")
    if goto_media:
        return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition.public_id))
    return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))


@app.route("/upload", methods=["GET"])
@login_required
def upload_page():
    """
    上傳頁面
    顯示上傳照片/影片的表單
    支援 exhibition_public_id 參數來關聯展覽（對外用 public_id）
    """
    exhibition_public_id = request.args.get("exhibition_public_id", "").strip()
    exhibition = None
    exhibitions = []
    
    if exhibition_public_id:
        exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id, is_published=True).first()
    
    # 獲取所有公開的展覽供選擇
    exhibitions = Exhibition.query.filter_by(is_published=True).order_by(Exhibition.created_at.desc()).all()
    
    return render_template("index.html", exhibition=exhibition, exhibitions=exhibitions)


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
    
    # 步驟 2.5：取得展覽（可選，表單可送 exhibition_public_id 或舊的 exhibition_id）
    exhibition_id = None
    pid = request.form.get("exhibition_public_id", "").strip()
    if pid:
        ex = Exhibition.query.filter_by(public_id=pid, is_published=True).first()
        if ex:
            exhibition_id = ex.id
    if exhibition_id is None:
        exhibition_id = request.form.get("exhibition_id", type=int)
        if exhibition_id:
            ex = db.session.get(Exhibition, exhibition_id)
            if not ex or not ex.is_published:
                exhibition_id = None
    
    # 步驟 3：檢查檔案格式
    original_filename = file.filename
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
        abort(400, "檔案格式不支援")

    # 步驟 4：暫時 ID，存檔後再換成 8+15位序號+4碼隨機
    media_id = _temp_media_id()
    
    # 按日期組織檔案：uploads/images/YYYY/MM/檔案名
    upload_date = datetime.now()
    if ext in ALLOWED_IMAGE_EXT:
        date_dir = UPLOAD_IMAGE_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
        date_dir.mkdir(parents=True, exist_ok=True)
        saved_path = date_dir / f"{media_id}{ext}"
        file_type = "image"
    else:
        date_dir = UPLOAD_VIDEO_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
        date_dir.mkdir(parents=True, exist_ok=True)
        saved_path = date_dir / f"{media_id}{ext}"
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
            exhibition_id=exhibition_id if exhibition_id else None,
        )
        db.session.add(media_record)
        db.session.commit()
        media_id = _apply_real_media_id(media_record)
        if media_record.exhibition_id:
            max_order = db.session.query(db.func.max(ExhibitionPhoto.display_order)).filter_by(
                exhibition_id=media_record.exhibition_id
            ).scalar() or -1
            up = Path(media_record.upload_path)
            photo_path_rel = up.relative_to(BASE_DIR) if up.is_absolute() else up
            preview_path = PREVIEW_DIR / f"{media_record.media_id}_preview.jpg"
            thumb_rel = str(preview_path.relative_to(BASE_DIR)) if preview_path.exists() else str(photo_path_rel)
            db.session.add(ExhibitionPhoto(
                exhibition_id=media_record.exhibition_id,
                photo_path=str(photo_path_rel),
                thumbnail_path=thumb_rel,
                title=media_record.original_filename or "",
                description="",
                display_order=max_order + 1,
                created_at=datetime.now()
            ))
        db.session.commit()
        if media_record.exhibition_id:
            ex = db.session.get(Exhibition, media_record.exhibition_id)
            if ex:
                return redirect(url_for("media_by_exhibition", exhibition_public_id=ex.public_id))
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
        exhibition_id=exhibition_id if exhibition_id else None,
    )
    db.session.add(media_record)
    db.session.commit()
    media_id = _apply_real_media_id(media_record)
    if media_record.exhibition_id:
        max_order = db.session.query(db.func.max(ExhibitionPhoto.display_order)).filter_by(
            exhibition_id=media_record.exhibition_id
        ).scalar() or -1
        up = Path(media_record.upload_path)
        photo_path_rel = up.relative_to(BASE_DIR) if up.is_absolute() else up
        preview_path = PREVIEW_DIR / f"{media_record.media_id}_preview.jpg"
        thumb_rel = str(preview_path.relative_to(BASE_DIR)) if preview_path.exists() else str(photo_path_rel)
        db.session.add(ExhibitionPhoto(
            exhibition_id=media_record.exhibition_id,
            photo_path=str(photo_path_rel),
            thumbnail_path=thumb_rel,
            title=media_record.original_filename or "",
            description="",
            display_order=max_order + 1,
            created_at=datetime.now()
        ))
    db.session.commit()
    if media_record.exhibition_id:
        ex = db.session.get(Exhibition, media_record.exhibition_id)
        if ex:
            return redirect(url_for("media_by_exhibition", exhibition_public_id=ex.public_id))
    return redirect(url_for("options", media_id=media_id))


@app.route("/result/<media_id>")
@login_required
def result(media_id):
    """
    顯示處理結果頁面
    從資料庫讀取已處理的媒體資料
    """
    # 從資料庫查詢媒體記錄
    # 超級管理員可以查看任何檔案，一般用戶只能查看自己的
    media = Media.query.filter_by(media_id=media_id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    # 權限檢查：超級管理員、媒體上傳者、或媒體所屬展覽的創辦人可以查看
    has_permission = False
    if current_user.is_super_admin_role():
        has_permission = True
    elif media.user_id == current_user.id:
        has_permission = True
    elif media.exhibition_id:
        exhibition = db.session.get(Exhibition, media.exhibition_id)
        if exhibition and current_user.can_manage_exhibition(exhibition):
            has_permission = True
    
    if not has_permission:
        abort(403, "您沒有權限查看此檔案")
    
    if media.status != "processed" or not media.output_path:
        abort(400, "檔案尚未處理完成")
    
    # 取得結果檔案 URL（支援新的日期目錄結構）
    is_video = media.file_type == "video"
    output_path = Path(media.output_path)
    
    # 計算相對路徑（從 OUTPUT_IMAGE_DIR 或 OUTPUT_VIDEO_DIR 開始）
    if is_video:
        try:
            rel_path = output_path.relative_to(OUTPUT_VIDEO_DIR)
            result_url = url_for("output_videos", filename=str(rel_path).replace("\\", "/"))
        except ValueError:
            # 如果路徑不在 OUTPUT_VIDEO_DIR 下，使用檔名
            result_url = url_for("output_videos", filename=output_path.name)
    else:
        try:
            rel_path = output_path.relative_to(OUTPUT_IMAGE_DIR)
            result_url = url_for("output_images", filename=str(rel_path).replace("\\", "/"))
        except ValueError:
            # 如果路徑不在 OUTPUT_IMAGE_DIR 下，使用檔名
            result_url = url_for("output_images", filename=output_path.name)
    
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

    # 優先從 DB 的 upload_path 取得路徑，避免大量檔案時 rglob 掃描整棵目錄樹
    media_record_for_path = Media.query.filter_by(media_id=media_id).first()
    if not media_record_for_path:
        abort(404, "找不到該檔案")
    
    # 權限檢查：超級管理員、媒體上傳者、或媒體所屬展覽的創辦人可以處理
    has_permission = False
    if current_user.is_super_admin_role():
        has_permission = True
    elif media_record_for_path.user_id == current_user.id:
        has_permission = True
    elif media_record_for_path.exhibition_id:
        exhibition = db.session.get(Exhibition, media_record_for_path.exhibition_id)
        if exhibition and current_user.can_manage_exhibition(exhibition):
            has_permission = True
    
    if not has_permission:
        abort(403, "您沒有權限處理此檔案")
    
    if media_record_for_path.upload_path:
        up = Path(media_record_for_path.upload_path)
        src_path = up if up.is_absolute() else BASE_DIR / up
        if not src_path.exists():
            src_path = None
    else:
        src_path = None
    if src_path is None:
        candidates = list(UPLOAD_IMAGE_DIR.rglob(f"{media_id}.*"))
        if not candidates:
            candidates = list(UPLOAD_VIDEO_DIR.rglob(f"{media_id}.*"))
        if not candidates:
            abort(404, "找不到檔案")
        src_path = candidates[0]

    overlay_file = request.files.get("overlay")
    overlay_path = None
    if overlay_file and overlay_file.filename:
        overlay_ext = Path(overlay_file.filename).suffix.lower()
        if overlay_ext not in ALLOWED_IMAGE_EXT:
            abort(400, "圖片格式不支援")
        # 查找原始檔案所在的目錄，將 overlay 保存在同一目錄
        overlay_dir = src_path.parent
        overlay_path = overlay_dir / f"{media_id}_overlay{overlay_ext}"
        overlay_file.save(overlay_path)

    # 使用模組化處理器處理媒體檔案
    try:
        # 建立處理器（使用預設靈敏度 0.6，可從 media_record 取得自訂值）
        processor = MediaProcessor(sensitivity=0.6)
        
        # 設定輸出路徑（按日期組織）
        upload_date = datetime.now()
        if _is_image(src_path):
            date_dir = OUTPUT_IMAGE_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
            date_dir.mkdir(parents=True, exist_ok=True)
            out_path = date_dir / f"{media_id}_out.jpg"
        else:
            date_dir = OUTPUT_VIDEO_DIR / str(upload_date.year) / f"{upload_date.month:02d}"
            date_dir.mkdir(parents=True, exist_ok=True)
            out_path = date_dir / f"{media_id}_out.mp4"
        
        # 處理媒體檔案
        output_path = processor.process(
            media_path=src_path,
            mode=mode,
            selected_face_ids=selected_ids if selected_ids else None,
            overlay_path=overlay_path,
            output_path=out_path,
        )
        
        # 更新資料庫記錄
        media_record = Media.query.filter_by(media_id=media_id).first()
        if media_record:
            media_record.output_path = str(output_path)
            media_record.process_mode = mode
            media_record.status = "processed"
            media_record.processed_at = datetime.now()
            # 編修（後續加隱私處理）：只重算 media_id 最後 4 碼並 rename 相關檔案
            if len(media_id) == 20 and media_id[0] == "8":
                new_media_id = _refresh_media_id_suffix(media_id)
                if new_media_id != media_id:
                    _rename_media_ids(media_id, new_media_id, media_record)
                    media_id = new_media_id
            
            # 如果媒體檔案有關聯到展覽：展覽應顯示「處理後」的檔案，故更新既有展覽照片為 output，或無對應時才新增
            if media_record.exhibition_id:
                # 使用 media_record.output_path（_rename_media_ids 可能已改檔名，須用更新後的路徑）
                output_path_for_display = Path(media_record.output_path)
                if not output_path_for_display.is_absolute():
                    output_path_for_display = BASE_DIR / output_path_for_display
                try:
                    relative_path = output_path_for_display.relative_to(BASE_DIR)
                except ValueError:
                    relative_path = output_path_for_display
                
                upload_full = Path(media_record.upload_path).resolve() if media_record.upload_path else None
                existing_photo = None
                if upload_full and upload_full.exists():
                    for ep in ExhibitionPhoto.query.filter_by(exhibition_id=media_record.exhibition_id).all():
                        ep_path = Path(ep.photo_path)
                        if not ep_path.is_absolute():
                            ep_path = BASE_DIR / ep_path
                        if ep_path.resolve() == upload_full:
                            existing_photo = ep
                            break
                
                if media_record.file_type == "video":
                    preview_files = list(PREVIEW_DIR.glob(f"{media_id}_preview.*"))
                    thumbnail_path = preview_files[0].relative_to(BASE_DIR) if preview_files else str(relative_path)
                else:
                    thumbnail_path = str(relative_path)
                
                if existing_photo:
                    # 更新既有展覽照片：改為顯示處理後的檔案（展覽不再顯示原檔）
                    existing_photo.photo_path = str(relative_path)
                    existing_photo.thumbnail_path = thumbnail_path
                    existing_photo.title = media_record.original_filename or (f"處理後的{'影片' if media_record.file_type == 'video' else '照片'} {media_id}")
                    existing_photo.description = f"處理模式: {mode}"
                else:
                    # 無對應的展覽照片（例如從選項頁關聯展覽）則新增一筆，直接使用處理後路徑
                    max_order = db.session.query(db.func.max(ExhibitionPhoto.display_order)).filter_by(
                        exhibition_id=media_record.exhibition_id
                    ).scalar() or -1
                    exhibition_photo = ExhibitionPhoto(
                        exhibition_id=media_record.exhibition_id,
                        photo_path=str(relative_path),
                        thumbnail_path=thumbnail_path,
                        title=media_record.original_filename or (f"處理後的{'影片' if media_record.file_type == 'video' else '照片'} {media_id}"),
                        description=f"處理模式: {mode}",
                        display_order=max_order + 1,
                        created_at=datetime.now()
                    )
                    db.session.add(exhibition_photo)
            
            db.session.commit()
        
        # 重定向到結果頁面
        return redirect(url_for("result", media_id=media_id))
    
    except ValueError as e:
        abort(400, str(e))
    except RuntimeError as e:
        abort(500, str(e))
    except Exception as e:
        abort(500, f"處理失敗: {str(e)}")


@app.route("/outputs/images/<path:filename>")
@login_required
def output_images(filename):
    """
    提供處理後的照片下載/顯示
    需要登入才能存取
    支援新的日期目錄結構：outputs/images/YYYY/MM/檔案名
    """
    # 查找檔案（支援日期目錄結構）
    candidates = list(OUTPUT_IMAGE_DIR.rglob(filename))
    if not candidates:
        abort(404, "檔案不存在")
    
    file_path = candidates[0]
    return send_from_directory(file_path.parent, file_path.name, as_attachment=False)


@app.route("/outputs/videos/<path:filename>")
@login_required
def output_videos(filename):
    """
    提供處理後的影片下載/播放
    需要登入才能存取
    支援新的日期目錄結構：outputs/videos/YYYY/MM/檔案名
    """
    # 查找檔案（支援日期目錄結構）
    candidates = list(OUTPUT_VIDEO_DIR.rglob(filename))
    if not candidates:
        abort(404, "檔案不存在")
    
    file_path = candidates[0]
    return send_from_directory(file_path.parent, file_path.name, as_attachment=False)


@app.route("/previews/<path:filename>")
@login_required
def previews(filename):
    """
    提供預覽圖片（含人臉框）
    需要登入才能存取
    """
    return send_from_directory(PREVIEW_DIR, filename, as_attachment=False)


@app.route("/uploads/images/<path:filename>")
@login_required
def upload_images(filename):
    """
    提供原始上傳的圖片
    需要登入才能存取
    支援新的日期目錄結構：uploads/images/YYYY/MM/檔案名
    """
    # 查找檔案（支援日期目錄結構）
    candidates = list(UPLOAD_IMAGE_DIR.rglob(filename))
    if not candidates:
        abort(404, "檔案不存在")
    
    file_path = candidates[0]
    return send_from_directory(file_path.parent, file_path.name, as_attachment=False)


@app.route("/uploads/videos/<path:filename>")
@login_required
def upload_videos(filename):
    """
    提供原始上傳的影片
    需要登入才能存取
    支援新的日期目錄結構：uploads/videos/YYYY/MM/檔案名
    """
    # 查找檔案（支援日期目錄結構）
    candidates = list(UPLOAD_VIDEO_DIR.rglob(filename))
    if not candidates:
        abort(404, "檔案不存在")
    
    file_path = candidates[0]
    return send_from_directory(file_path.parent, file_path.name, as_attachment=False)


@app.route("/exhibition/<exhibition_public_id>/photo/<int:photo_id>/delete", methods=["POST"])
@login_required
def delete_exhibition_photo(exhibition_public_id, photo_id):
    """
    刪除展覽照片（對外使用 public_id）
    權限規則：
    - 超級管理員可以刪除任何照片
    - 展覽創建者可以刪除展覽中的任何照片
    - 照片上傳者可以刪除自己上傳的照片（通過 Media 記錄查找）
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    photo = ExhibitionPhoto.query.get_or_404(photo_id)
    
    # 檢查照片是否屬於該展覽
    if photo.exhibition_id != exhibition.id:
        abort(404, "照片不存在")
    
    # 權限檢查
    has_permission = False
    
    # 1. 超級管理員可以刪除任何照片
    if current_user.is_super_admin_role():
        has_permission = True
    # 2. 展覽創建者可以刪除展覽中的任何照片
    elif current_user.can_manage_exhibition(exhibition):
        has_permission = True
    # 3. 檢查是否是照片上傳者（通過 Media 記錄查找）
    else:
        # 嘗試通過 photo_path 找到對應的 Media 記錄
        photo_path = photo.photo_path
        photo_path_obj = Path(photo_path)
        if photo_path_obj.is_absolute():
            full_photo_path = photo_path_obj
        else:
            full_photo_path = BASE_DIR / photo_path_obj
        
        # 獲取檔案名稱用於匹配
        photo_filename = full_photo_path.name
        
        # 查找對應的 Media 記錄
        media = Media.query.filter(
            (Media.upload_path.like(f"%{photo_filename}")) |
            (Media.output_path.like(f"%{photo_filename}")) |
            (Media.upload_path == str(photo_path)) |
            (Media.upload_path == str(full_photo_path)) |
            (Media.output_path == str(photo_path)) |
            (Media.output_path == str(full_photo_path))
        ).first()
        
        # 如果是照片上傳者，可以刪除
        if media and media.user_id == current_user.id:
            has_permission = True
    
    if not has_permission:
        abort(403, "您沒有權限刪除此照片")
    
    # 找出對應的 Media 記錄（若存在），刪除展覽照片時一併刪除，媒體管理才不會殘留
    photo_path_str = photo.photo_path
    photo_path_for_lookup = Path(photo_path_str)
    full_photo_path_lookup = photo_path_for_lookup if photo_path_for_lookup.is_absolute() else BASE_DIR / photo_path_for_lookup
    photo_filename = full_photo_path_lookup.name
    linked_media = Media.query.filter(
        (Media.upload_path.like(f"%{photo_filename}")) |
        (Media.output_path.like(f"%{photo_filename}")) |
        (Media.upload_path == photo_path_str) |
        (Media.upload_path == str(full_photo_path_lookup)) |
        (Media.output_path == photo_path_str) |
        (Media.output_path == str(full_photo_path_lookup))
    ).first()
    
    errors = []
    
    # 刪除檔案（如果存在）
    photo_path = Path(photo.photo_path)
    if photo_path.is_absolute():
        full_path = photo_path
    else:
        full_path = BASE_DIR / photo_path
    
    if full_path.exists():
        try:
            full_path.unlink()
        except Exception as e:
            errors.append(f"無法刪除檔案: {e}")
    
    # 刪除縮圖（如果不同於主檔案且存在）
    if photo.thumbnail_path and photo.thumbnail_path != photo.photo_path:
        thumb_path = Path(photo.thumbnail_path)
        if thumb_path.is_absolute():
            full_thumb_path = thumb_path
        else:
            full_thumb_path = BASE_DIR / thumb_path
        
        if full_thumb_path.exists():
            try:
                full_thumb_path.unlink()
            except Exception as e:
                errors.append(f"無法刪除縮圖: {e}")
    
    # 若有對應 Media，一併刪除其上傳/處理檔與 preview、人臉相關檔案
    if linked_media:
        media_id = linked_media.media_id
        if linked_media.upload_path:
            up = Path(linked_media.upload_path)
            if not up.is_absolute():
                up = BASE_DIR / up
            if up.exists():
                try:
                    up.unlink()
                except Exception as e:
                    errors.append(f"無法刪除上傳檔案: {e}")
        if linked_media.output_path:
            op = Path(linked_media.output_path)
            if not op.is_absolute():
                op = BASE_DIR / op
            if op.exists():
                try:
                    op.unlink()
                except Exception as e:
                    errors.append(f"無法刪除處理檔案: {e}")
        try:
            for preview_file in PREVIEW_DIR.glob(f"{media_id}_preview.*"):
                if preview_file.exists():
                    try:
                        preview_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除預覽圖 {preview_file.name}: {e}")
        except Exception as e:
            errors.append(f"查找預覽圖時出錯: {e}")
        try:
            for crop_file in PREVIEW_DIR.glob(f"{media_id}_face_*.jpg"):
                if crop_file.exists():
                    try:
                        crop_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除人臉截圖 {crop_file.name}: {e}")
        except Exception as e:
            errors.append(f"查找人臉截圖時出錯: {e}")
        faces_json = METADATA_DIR / f"{media_id}_faces.json"
        if faces_json.exists():
            try:
                faces_json.unlink()
            except Exception as e:
                errors.append(f"無法刪除人臉資料: {e}")
    
    # 刪除資料庫記錄（ExhibitionPhoto 與對應的 Media，媒體管理才不會殘留）
    try:
        db.session.delete(photo)
        if linked_media:
            db.session.delete(linked_media)
        db.session.commit()
        
        if errors:
            flash(f"照片已刪除，但部分檔案刪除失敗: {'; '.join(errors)}", "warning")
        else:
            flash(_("照片已成功刪除"), "success")
    except Exception as e:
        db.session.rollback()
        flash(f"刪除失敗：{str(e)}", "error")
    
    return redirect(url_for("exhibition_detail", exhibition_public_id=exhibition.public_id))


# ==================== 媒體管理 ====================

@app.route("/media")
@login_required
def media_list():
    """
    媒體檔案管理頁面
    顯示當前用戶有媒體檔案的展覽列表
    超級管理員可以看到所有展覽
    """
    # 統計每個展覽的媒體數量
    exhibition_stats = {}
    uncategorized_count = 0
    
    if current_user.is_super_admin_role():
        # 超級管理員可以看到所有媒體檔案
        all_media = Media.query.all()
        for media in all_media:
            if media.exhibition_id:
                if media.exhibition_id not in exhibition_stats:
                    exhibition = db.session.get(Exhibition, media.exhibition_id)
                    if exhibition:
                        exhibition_stats[media.exhibition_id] = {
                            'exhibition': exhibition,
                            'count': 0
                        }
                if media.exhibition_id in exhibition_stats:
                    exhibition_stats[media.exhibition_id]['count'] += 1
            else:
                uncategorized_count += 1
    else:
        # 一般用戶：統計自己上傳的媒體 + 自己創辦的展覽中的所有媒體
        # 1. 先找出所有需要顯示的展覽：自己上傳的媒體所屬展覽 + 自己創辦的展覽
        own_media = Media.query.filter_by(user_id=current_user.id).all()
        exhibitions_to_show = set()
        
        for media in own_media:
            if media.exhibition_id:
                exhibitions_to_show.add(media.exhibition_id)
            else:
                uncategorized_count += 1
        
        # 加入自己創辦的展覽
        own_exhibitions = Exhibition.query.filter_by(creator_id=current_user.id).all()
        for exhibition in own_exhibitions:
            exhibitions_to_show.add(exhibition.id)
        
        # 2. 對每個展覽統計媒體數量
        for ex_id in exhibitions_to_show:
            exhibition = db.session.get(Exhibition, ex_id)
            if not exhibition:
                continue
            
            # 如果是創辦人，統計該展覽的所有媒體；否則只統計自己的媒體
            if exhibition.creator_id == current_user.id:
                media_count = Media.query.filter_by(exhibition_id=ex_id).count()
            else:
                media_count = Media.query.filter_by(
                    exhibition_id=ex_id,
                    user_id=current_user.id
                ).count()
            
            if media_count > 0:
                exhibition_stats[ex_id] = {
                    'exhibition': exhibition,
                    'count': media_count
                }
    
    # 轉換為列表並排序
    exhibitions_with_media = list(exhibition_stats.values())
    exhibitions_with_media.sort(key=lambda x: x['exhibition'].created_at, reverse=True)
    
    return render_template("media_list.html", 
                          exhibitions_with_media=exhibitions_with_media,
                          uncategorized_count=uncategorized_count)


@app.route("/media/exhibition/<exhibition_public_id>")
@login_required
def media_by_exhibition(exhibition_public_id):
    """
    顯示特定展覽的媒體檔案（對外使用 public_id）
    超級管理員和展覽創辦人可以看到該展覽的所有媒體檔案，一般用戶只能看到自己的
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    if current_user.is_super_admin_role() or current_user.can_manage_exhibition(exhibition):
        # 超級管理員或展覽創辦人可以看到該展覽的所有媒體
        media_list = Media.query.filter_by(
            exhibition_id=exhibition.id
        ).order_by(Media.created_at.desc()).all()
    else:
        # 一般用戶只能看到自己上傳的媒體
        media_list = Media.query.filter_by(
            user_id=current_user.id,
            exhibition_id=exhibition.id
        ).order_by(Media.created_at.desc()).all()
    
    # 載入 user 關聯（用於顯示上傳者資訊）並計算預覽圖 URL
    for media in media_list:
        if media.user_id:
            media.user = db.session.get(User, media.user_id)
        
        # 計算預覽圖 URL
        preview_filename = f"{media.media_id}_preview.jpg"
        preview_path = PREVIEW_DIR / preview_filename
        if preview_path.exists():
            # 如果預覽圖存在，使用預覽圖
            media.preview_url = url_for("previews", filename=preview_filename)
        elif media.upload_path:
            # 如果預覽圖不存在，使用原始圖片/影片
            upload_path = Path(media.upload_path)
            if not upload_path.exists():
                # 如果檔案不存在，使用預設預覽圖
                media.preview_url = url_for("previews", filename=preview_filename)
            else:
                # 構建相對路徑（從 UPLOAD_IMAGE_DIR 或 UPLOAD_VIDEO_DIR 開始）
                if media.file_type == "image":
                    try:
                        # 嘗試從 UPLOAD_IMAGE_DIR 計算相對路徑
                        upload_rel = upload_path.relative_to(UPLOAD_IMAGE_DIR)
                        media.preview_url = url_for("upload_images", filename=str(upload_rel).replace("\\", "/"))
                    except ValueError:
                        # 如果路徑不在 UPLOAD_IMAGE_DIR 下，嘗試從 BASE_DIR 計算
                        try:
                            rel_path = upload_path.relative_to(BASE_DIR)
                            media.preview_url = url_for("upload_images", filename=str(rel_path).replace("\\", "/"))
                        except ValueError:
                            # 如果都不行，使用檔名
                            media.preview_url = url_for("upload_images", filename=upload_path.name)
                else:
                    try:
                        # 嘗試從 UPLOAD_VIDEO_DIR 計算相對路徑
                        upload_rel = upload_path.relative_to(UPLOAD_VIDEO_DIR)
                        media.preview_url = url_for("upload_videos", filename=str(upload_rel).replace("\\", "/"))
                    except ValueError:
                        # 如果路徑不在 UPLOAD_VIDEO_DIR 下，嘗試從 BASE_DIR 計算
                        try:
                            rel_path = upload_path.relative_to(BASE_DIR)
                            media.preview_url = url_for("upload_videos", filename=str(rel_path).replace("\\", "/"))
                        except ValueError:
                            # 如果都不行，使用檔名
                            media.preview_url = url_for("upload_videos", filename=upload_path.name)
        else:
            # 如果沒有上傳路徑，使用預設預覽圖
            media.preview_url = url_for("previews", filename=preview_filename)
    
    return render_template("media_by_exhibition.html", 
                          exhibition=exhibition,
                          media_list=media_list)


@app.route("/media/uncategorized")
@login_required
def media_uncategorized():
    """
    顯示未分類的媒體檔案（沒有關聯展覽的）
    超級管理員可以看到所有未分類的媒體檔案，一般用戶只能看到自己的
    """
    # 超級管理員可以看到所有未分類的媒體檔案，一般用戶只能看到自己的
    if current_user.is_super_admin_role():
        media_list = Media.query.filter_by(
            exhibition_id=None
        ).order_by(Media.created_at.desc()).all()
    else:
        media_list = Media.query.filter_by(
            user_id=current_user.id,
            exhibition_id=None
        ).order_by(Media.created_at.desc()).all()
    
    # 載入 user 關聯（用於顯示上傳者資訊）並計算預覽圖 URL
    for media in media_list:
        if media.user_id:
            media.user = db.session.get(User, media.user_id)
        
        # 計算預覽圖 URL
        preview_filename = f"{media.media_id}_preview.jpg"
        preview_path = PREVIEW_DIR / preview_filename
        if preview_path.exists():
            # 如果預覽圖存在，使用預覽圖
            media.preview_url = url_for("previews", filename=preview_filename)
        elif media.upload_path:
            # 如果預覽圖不存在，使用原始圖片/影片
            upload_path = Path(media.upload_path)
            if not upload_path.exists():
                # 如果檔案不存在，使用預設預覽圖
                media.preview_url = url_for("previews", filename=preview_filename)
            else:
                # 構建相對路徑（從 UPLOAD_IMAGE_DIR 或 UPLOAD_VIDEO_DIR 開始）
                if media.file_type == "image":
                    try:
                        # 嘗試從 UPLOAD_IMAGE_DIR 計算相對路徑
                        upload_rel = upload_path.relative_to(UPLOAD_IMAGE_DIR)
                        media.preview_url = url_for("upload_images", filename=str(upload_rel).replace("\\", "/"))
                    except ValueError:
                        # 如果路徑不在 UPLOAD_IMAGE_DIR 下，嘗試從 BASE_DIR 計算
                        try:
                            rel_path = upload_path.relative_to(BASE_DIR)
                            media.preview_url = url_for("upload_images", filename=str(rel_path).replace("\\", "/"))
                        except ValueError:
                            # 如果都不行，使用檔名
                            media.preview_url = url_for("upload_images", filename=upload_path.name)
                else:
                    try:
                        # 嘗試從 UPLOAD_VIDEO_DIR 計算相對路徑
                        upload_rel = upload_path.relative_to(UPLOAD_VIDEO_DIR)
                        media.preview_url = url_for("upload_videos", filename=str(upload_rel).replace("\\", "/"))
                    except ValueError:
                        # 如果路徑不在 UPLOAD_VIDEO_DIR 下，嘗試從 BASE_DIR 計算
                        try:
                            rel_path = upload_path.relative_to(BASE_DIR)
                            media.preview_url = url_for("upload_videos", filename=str(rel_path).replace("\\", "/"))
                        except ValueError:
                            # 如果都不行，使用檔名
                            media.preview_url = url_for("upload_videos", filename=upload_path.name)
        else:
            # 如果沒有上傳路徑，使用預設預覽圖
            media.preview_url = url_for("previews", filename=preview_filename)
    
    return render_template("media_by_exhibition.html", 
                          exhibition=None,
                          media_list=media_list)


def _delete_media_file(media_id):
    """
    刪除單個媒體檔案的輔助函數
    返回 (success, error_message)
    """
    try:
        media = Media.query.filter_by(media_id=media_id).first()
        if not media:
            return False, f"找不到檔案 {media_id}"
        
        # 權限檢查：超級管理員、媒體上傳者、或媒體所屬展覽的創辦人可以刪除
        has_permission = False
        if current_user.is_super_admin_role():
            has_permission = True
        elif media.user_id == current_user.id:
            has_permission = True
        elif media.exhibition_id:
            exhibition = db.session.get(Exhibition, media.exhibition_id)
            if exhibition and current_user.can_manage_exhibition(exhibition):
                has_permission = True
        
        if not has_permission:
            return False, f"沒有權限刪除檔案 {media_id}"
        
        errors = []
        
        # 刪除原始上傳檔案（路徑可能為相對，需依 BASE_DIR 解析）
        if media.upload_path:
            upload_file = Path(media.upload_path)
            if not upload_file.is_absolute():
                upload_file = BASE_DIR / upload_file
            if upload_file.exists():
                try:
                    upload_file.unlink()
                except Exception as e:
                    errors.append(f"無法刪除上傳檔案: {e}")
        
        # 刪除處理後的檔案（路徑可能為相對，需依 BASE_DIR 解析）
        if media.output_path:
            output_file = Path(media.output_path)
            if not output_file.is_absolute():
                output_file = BASE_DIR / output_file
            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception as e:
                    errors.append(f"無法刪除處理檔案: {e}")
        
        # 刪除預覽圖（使用 glob 匹配所有可能的副檔名）
        try:
            preview_files = list(PREVIEW_DIR.glob(f"{media_id}_preview.*"))
            for preview_file in preview_files:
                if preview_file.exists():
                    try:
                        preview_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除預覽圖 {preview_file.name}: {e}")
        except Exception as e:
            errors.append(f"查找預覽圖時出錯: {e}")
        
        # 刪除人臉資料 JSON
        faces_json = METADATA_DIR / f"{media_id}_faces.json"
        if faces_json.exists():
            try:
                faces_json.unlink()
            except Exception as e:
                errors.append(f"無法刪除人臉資料: {e}")
        
        # 刪除人臉截圖（注意：人臉截圖保存在 PREVIEW_DIR，不是 METADATA_DIR）
        try:
            face_crops_preview = list(PREVIEW_DIR.glob(f"{media_id}_face_*.jpg"))
            for crop_file in face_crops_preview:
                if crop_file.exists():
                    try:
                        crop_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除人臉截圖 {crop_file.name}: {e}")
        except Exception as e:
            errors.append(f"查找人臉截圖時出錯: {e}")
        
        # 也檢查 METADATA_DIR（以防萬一）
        try:
            face_crops_metadata = list(METADATA_DIR.glob(f"{media_id}_face_*.jpg"))
            for crop_file in face_crops_metadata:
                if crop_file.exists():
                    try:
                        crop_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除人臉截圖 {crop_file.name}: {e}")
        except Exception as e:
            pass  # METADATA_DIR 中沒有人臉截圖是正常的
        
        # 刪除資料庫記錄
        db.session.delete(media)
        
        # 如果有錯誤但資料庫記錄已刪除，仍然返回成功（檔案可能已經不存在）
        if errors:
            return True, f"部分檔案刪除失敗: {'; '.join(errors)}"
        
        return True, None
    except Exception as e:
        return False, str(e)


@app.route("/media/<media_id>/delete", methods=["POST"])
@login_required
def delete_media(media_id):
    """
    刪除媒體檔案
    超級管理員可以刪除任何檔案，一般用戶只能刪除自己的檔案
    """
    media = Media.query.filter_by(media_id=media_id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    # 權限檢查：超級管理員、媒體上傳者、或媒體所屬展覽的創辦人可以刪除
    has_permission = False
    if current_user.is_super_admin_role():
        has_permission = True
    elif media.user_id == current_user.id:
        has_permission = True
    elif media.exhibition_id:
        exhibition = db.session.get(Exhibition, media.exhibition_id)
        if exhibition and current_user.can_manage_exhibition(exhibition):
            has_permission = True
    
    if not has_permission:
        abort(403, "您沒有權限刪除此檔案")
    
    # 保存展覽（用於重定向，對外使用 public_id）
    exhibition = db.session.get(Exhibition, media.exhibition_id) if media.exhibition_id else None
    
    success, error = _delete_media_file(media_id)
    
    if success:
        db.session.commit()
        flash("檔案已成功刪除", "success")
    else:
        db.session.rollback()
        flash(f"刪除失敗：{error}", "error")
    
    if exhibition:
        return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition.public_id))
    return redirect(url_for("media_uncategorized"))


@app.route("/media/bulk-delete", methods=["POST"])
@login_required
def bulk_delete_media():
    """
    批量刪除媒體檔案
    超級管理員可以刪除任何檔案，一般用戶只能刪除自己的檔案
    """
    media_ids_str = request.form.get("media_ids", "")
    exhibition_public_id = request.form.get("exhibition_public_id", "").strip()
    if not exhibition_public_id:
        ex_id = request.form.get("exhibition_id", type=int)
        if ex_id:
            ex = db.session.get(Exhibition, ex_id)
            exhibition_public_id = ex.public_id if ex else ""
    
    if not media_ids_str:
        flash("未選擇任何檔案", "error")
        if exhibition_public_id:
            return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition_public_id))
        return redirect(url_for("media_uncategorized"))
    
    media_ids = [mid.strip() for mid in media_ids_str.split(",") if mid.strip()]
    
    if not media_ids:
        flash("未選擇任何檔案", "error")
        if exhibition_public_id:
            return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition_public_id))
        return redirect(url_for("media_uncategorized"))
    
    success_count = 0
    error_count = 0
    errors = []
    
    for media_id in media_ids:
        success, error = _delete_media_file(media_id)
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(error)
    
    # 提交所有成功的刪除
    if success_count > 0:
        db.session.commit()
    
    # 顯示結果訊息
    if error_count == 0:
        flash(f"成功刪除 {success_count} 個檔案", "success")
    elif success_count == 0:
        db.session.rollback()
        flash(f"刪除失敗：{errors[0] if errors else '未知錯誤'}", "error")
    else:
        flash(f"成功刪除 {success_count} 個檔案，{error_count} 個失敗", "error")
    
    if exhibition_public_id:
        return redirect(url_for("media_by_exhibition", exhibition_public_id=exhibition_public_id))
    return redirect(url_for("media_uncategorized"))


@app.route("/media/<media_id>/download")
@login_required
def download_media(media_id):
    """
    下載原始媒體檔案
    超級管理員可以下載任何檔案，一般用戶只能下載自己的檔案
    """
    media = Media.query.filter_by(media_id=media_id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    # 權限檢查：超級管理員可以下載任何檔案，一般用戶只能下載自己的
    if not current_user.is_super_admin_role() and media.user_id != current_user.id:
        abort(403, "您沒有權限下載此檔案")
    
    if not media.upload_path:
        abort(404, "檔案路徑不存在")
    
    file_path = Path(media.upload_path)
    if not file_path.exists():
        abort(404, "檔案不存在")
    
    return send_from_directory(file_path.parent, file_path.name, as_attachment=True, download_name=media.original_filename or file_path.name)


@app.route("/media/<media_id>/download_output")
@login_required
def download_output(media_id):
    """
    下載處理後的媒體檔案
    超級管理員可以下載任何檔案，一般用戶只能下載自己的檔案
    """
    media = Media.query.filter_by(media_id=media_id).first()
    if not media:
        abort(404, "找不到該檔案")
    
    # 權限檢查：超級管理員可以下載任何檔案，一般用戶只能下載自己的
    if not current_user.is_super_admin_role() and media.user_id != current_user.id:
        abort(403, "您沒有權限下載此檔案")
    
    if not media.output_path or media.status != "processed":
        abort(400, "檔案尚未處理完成")
    
    file_path = Path(media.output_path)
    if not file_path.exists():
        abort(404, "檔案不存在")
    
    # 生成下載檔名
    original_name = Path(media.original_filename).stem if media.original_filename else "output"
    ext = file_path.suffix
    download_name = f"{original_name}_processed{ext}"
    
    return send_from_directory(file_path.parent, file_path.name, as_attachment=True, download_name=download_name)


# ==================== 啟動應用程式 ====================

if __name__ == "__main__":
    # 開發模式：在本機 5000 port 啟動，並開啟除錯模式
    app.run(host="0.0.0.0", port=5000, debug=True)
