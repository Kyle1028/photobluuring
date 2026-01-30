"""
管理員功能模組
處理展覽管理、用戶角色管理等功能
"""

from datetime import datetime, date
import json
import re
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from flask_babel import gettext as _
from pathlib import Path
from werkzeug.utils import secure_filename

from core.models import (
    db,
    User,
    Exhibition,
    ExhibitionPhoto,
    Media,
    ExhibitionFloor,
    ExhibitionCell,
    ExhibitionMergedRegion,
    _public_id_exhibition,
    media_cells,
)
from core.decorators import admin_required, super_admin_required, can_manage_exhibition

# 建立管理員藍圖，所有管理員相關的路由都以 /admin 開頭
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# 目錄設定（避免循環導入，直接在此定義）
BASE_DIR = Path(__file__).resolve().parent.parent  # 專案根目錄（core 的上一層）
EXHIBITION_DIR = BASE_DIR / "exhibitions"  # 展覽照片目錄
PREVIEW_DIR = BASE_DIR / "previews"  # 預覽圖
METADATA_DIR = BASE_DIR / "metadata"  # 人臉資料


def init_admin(app):
    """
    初始化管理員系統
    在 Flask 應用程式啟動時呼叫此函式
    """
    app.register_blueprint(admin_bp)


def _generate_cells_for_floor(floor: ExhibitionFloor) -> None:
    """
    根據樓層的實際尺寸與 grid_size 為該樓層產生 ExhibitionCell。
    每層的 cell_code 從 C000001 重新編號，依「上到下、左到右」順序。
    """
    if not floor.width_meters or not floor.height_meters or not floor.grid_size:
        return

    cols = int(floor.width_meters / floor.grid_size)
    rows = int(floor.height_meters / floor.grid_size)

    # 獲取該樓層的所有現有 cells
    existing_cells = ExhibitionCell.query.filter_by(floor_id=floor.id).all()
    
    if existing_cells:
        # 先刪除該樓層的合併區（合併區底下的 cells 即將被刪除）
        ExhibitionMergedRegion.query.filter_by(floor_id=floor.id).delete()
        # 再刪除 media_cells 關聯表中的記錄（避免外鍵約束錯誤）
        cell_ids = [cell.id for cell in existing_cells]
        if cell_ids:
            db.session.execute(
                media_cells.delete().where(media_cells.c.cell_id.in_(cell_ids))
            )
        # 然後刪除 cells
        for cell in existing_cells:
            db.session.delete(cell)
        db.session.flush()  # 確保刪除操作已執行

    cells = []
    idx = 1
    for row in range(rows):
        for col in range(cols):
            code = f"C{idx:06d}"
            cells.append(
                ExhibitionCell(
                    floor_id=floor.id,
                    cell_code=code,
                    row=row,
                    col=col,
                    name=f"區域 {code}",
                    is_active=True,
                )
            )
            idx += 1

    if cells:
        db.session.add_all(cells)


def _apply_selection_polygon(floor: ExhibitionFloor, polygon_data) -> None:
    """
    根據圈選的多邊形區域，更新網格的 is_active 狀態。
    支持單個多邊形或多個區域的列表。
    
    Args:
        floor: ExhibitionFloor 物件
        polygon_data: 可以是：
            - 多邊形頂點列表，每個頂點為 {x: 0.0-1.0, y: 0.0-1.0}（相對於圖片尺寸的比例）
            - 區域列表，每個區域為 {type: 'rect'|'polygon', points: [...]}
    """
    if not polygon_data:
        return
    
    # 獲取該樓層的所有網格
    cells = ExhibitionCell.query.filter_by(floor_id=floor.id).all()
    if not cells:
        return
    
    # 計算網格數量
    cols = int(floor.width_meters / floor.grid_size)
    rows = int(floor.height_meters / floor.grid_size)
    
    # 判斷點是否在多邊形內（射線法）
    def point_in_polygon(point, polygon):
        inside = False
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            xi, yi = polygon[i]["x"], polygon[i]["y"]
            xj, yj = polygon[j]["x"], polygon[j]["y"]
            intersect = ((yi > point["y"]) != (yj > point["y"])) and \
                       (point["x"] < (xj - xi) * (point["y"] - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside
        return inside
    
    # 處理多個區域的情況
    areas = []
    if isinstance(polygon_data, list) and len(polygon_data) > 0:
        # 檢查第一個元素是否有 'type' 和 'points' 鍵（多個區域）
        if isinstance(polygon_data[0], dict) and 'type' in polygon_data[0] and 'points' in polygon_data[0]:
            # 多個區域的情況
            areas = polygon_data
        else:
            # 單個多邊形的情況（向後兼容）
            areas = [{"type": "polygon", "points": polygon_data}]
    
    if not areas:
        return
    
    # 將所有區域轉換為實際座標
    actual_polygons = []
    for area in areas:
        points = area.get("points", [])
        if len(points) < 3:
            continue
        actual_polygon = [
            {"x": p["x"] * floor.width_meters, "y": p["y"] * floor.height_meters}
            for p in points
        ]
        actual_polygons.append(actual_polygon)
    
    if not actual_polygons:
        return
    
    # 更新每個網格的 is_active 狀態
    # 只要網格中心點在任何一個區域內，就標記為有效
    for cell in cells:
        # 計算網格中心點的實際座標（公尺）
        center_x = (cell.col + 0.5) * floor.grid_size
        center_y = (cell.row + 0.5) * floor.grid_size
        
        # 檢查中心點是否在任何一個區域內
        is_inside = False
        for actual_polygon in actual_polygons:
            if point_in_polygon({"x": center_x, "y": center_y}, actual_polygon):
                is_inside = True
                break
        
        cell.is_active = is_inside


def _get_exhibition_dir(exhibition: Exhibition) -> Path:
    """
    取得該展覽的實體資料夾（用於封面/樓層檔案存放）。
    - 若已有 cover_image，沿用其資料夾
    - 否則以 title 產生安全目錄名稱
    """
    if exhibition.cover_image:
        p = BASE_DIR / exhibition.cover_image
        return p.parent
    safe_title = "".join(c for c in (exhibition.title or "") if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_title = safe_title.replace(" ", "_")[:50] or f"exhibition_{exhibition.id}"
    return EXHIBITION_DIR / safe_title


def _suggest_next_floor_code(exhibition: Exhibition) -> str:
    """
    根據既有樓層代碼，建議下一個 F00x。
    若找不到任何符合格式的樓層，預設回傳 F001。
    """
    max_n = 0
    for f in getattr(exhibition, "floors", []) or []:
        if not getattr(f, "floor_code", None):
            continue
        m = re.match(r"^F(\d{3})$", str(f.floor_code).upper())
        if not m:
            continue
        try:
            n = int(m.group(1))
            max_n = max(max_n, n)
        except ValueError:
            continue
    return f"F{max_n + 1:03d}" if max_n >= 0 else "F001"


# ==================== 展覽管理 ====================

@admin_bp.route("/exhibitions")
@admin_required
def exhibitions_list():
    """
    展覽管理列表頁面
    顯示所有展覽（管理員只能看到自己創建的，超級管理員可以看到所有）
    """
    if current_user.is_super_admin_role():
        # 超級管理員可以看到所有展覽
        exhibitions = Exhibition.query.order_by(Exhibition.created_at.desc()).all()
    else:
        # 一般管理員只能看到自己創建的展覽
        exhibitions = Exhibition.query.filter_by(creator_id=current_user.id).order_by(Exhibition.created_at.desc()).all()
    
    return render_template("admin/exhibitions_list.html", exhibitions=exhibitions)


@admin_bp.route("/exhibitions/create", methods=["GET", "POST"])
@admin_required
def create_exhibition():
    """
    創建展覽
    GET：顯示創建表單
    POST：處理表單並創建展覽
    """
    if request.method == "POST":
        # 取得表單資料
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        start_date_str = request.form.get("start_date", "").strip()
        end_date_str = request.form.get("end_date", "").strip()
        is_published = request.form.get("is_published") == "on"
        
        # 驗證必填欄位
        if not title:
            flash("請輸入展覽標題", "error")
            return render_template("admin/exhibition_form.html", mode="create")
        
        # 處理日期
        start_date = None
        end_date = None
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            except ValueError:
                flash("開始日期格式不正確", "error")
                return render_template("admin/exhibition_form.html", mode="create")
        
        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            except ValueError:
                flash("結束日期格式不正確", "error")
                return render_template("admin/exhibition_form.html", mode="create")
        
        # 處理封面圖片與樓層目錄
        cover_image_path = None
        cover_file = request.files.get("cover_image")
        # 先計算安全的展覽目錄名稱，供封面與樓層共用
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_title = safe_title.replace(" ", "_")[:50]  # 限制長度
        exhibition_dir = EXHIBITION_DIR / safe_title
        if cover_file and cover_file.filename:
            # 生成安全的檔案名稱
            filename = secure_filename(cover_file.filename)
            ext = Path(filename).suffix.lower()
            
            # 只允許圖片格式
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                flash("封面圖片格式不支援，請使用 JPG、PNG 或 WEBP", "error")
                return render_template("admin/exhibition_form.html", mode="create")
            
            # 創建展覽專用目錄（使用標題的簡化版本）
            exhibition_dir.mkdir(parents=True, exist_ok=True)
            
            # 儲存封面圖片
            cover_filename = f"cover{ext}"
            cover_image_path = exhibition_dir / cover_filename
            cover_file.save(cover_image_path)
            cover_image_path = str(cover_image_path.relative_to(BASE_DIR))
        
        # 創建展覽記錄
        try:
            exhibition = Exhibition(
                title=title,
                description=description,
                cover_image=cover_image_path,
                start_date=start_date,
                end_date=end_date,
                is_published=is_published,
                creator_id=current_user.id,
                created_at=datetime.now()
            )
            
            db.session.add(exhibition)
            db.session.commit()
            exhibition.public_id = _public_id_exhibition(exhibition.id)  # 7+9位序號+6碼隨機
            db.session.commit()

            # 若有提供第一層樓層平面圖與尺寸，建立 F001 與區域
            floor_file = request.files.get("floor_image_f001")
            width_str = request.form.get("floor_width_f001", "").strip()
            height_str = request.form.get("floor_height_f001", "").strip()
            grid_str = request.form.get("floor_grid_f001", "").strip()
            if floor_file and floor_file.filename and width_str and height_str:
                try:
                    width_m = float(width_str)
                    height_m = float(height_str)
                    grid_size = float(grid_str) if grid_str else 1.0
                except ValueError:
                    width_m = height_m = 0
                    grid_size = 1.0

                if width_m > 0 and height_m > 0:
                    filename = secure_filename(floor_file.filename)
                    ext = Path(filename).suffix.lower()
                    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                        flash("樓層平面圖格式不支援，請使用 JPG、PNG 或 WEBP", "error")
                        return render_template("admin/exhibition_form.html", mode="create")

                    exhibition_dir.mkdir(parents=True, exist_ok=True)
                    floor_filename = f"F001_floor{ext}"
                    floor_path = exhibition_dir / floor_filename
                    floor_file.save(floor_path)
                    floor_rel = str(floor_path.relative_to(BASE_DIR))

                    floor = ExhibitionFloor(
                        exhibition_id=exhibition.id,
                        floor_code="F001",
                        image_path=floor_rel,
                        width_meters=width_m,
                        height_meters=height_m,
                        grid_size=grid_size if grid_size > 0 else 1.0,
                        created_at=datetime.now(),
                    )
                    db.session.add(floor)
                    db.session.flush()  # 取得 floor.id 以產生 cells
                    _generate_cells_for_floor(floor)
                    
                    # 處理圈選區域（如果有）
                    selection_polygon_str = request.form.get("selection_polygon", "").strip()
                    if selection_polygon_str:
                        try:
                            polygon_data = json.loads(selection_polygon_str)
                            if polygon_data and len(polygon_data) >= 3:
                                _apply_selection_polygon(floor, polygon_data)
                        except (json.JSONDecodeError, ValueError) as e:
                            # 如果解析失敗，忽略錯誤，不影響展覽創建
                            pass
                    
                    db.session.commit()

            flash("展覽創建成功！", "success")
            return redirect(url_for("admin.exhibitions_list"))
        
        except Exception as e:
            db.session.rollback()
            flash(f"創建展覽失敗：{str(e)}", "error")
            return render_template("admin/exhibition_form.html", mode="create")
    
    # GET 請求：顯示創建表單
    return render_template("admin/exhibition_form.html", mode="create")


@admin_bp.route("/exhibitions/<exhibition_public_id>/edit", methods=["GET", "POST"])
@can_manage_exhibition('exhibition_public_id')
def edit_exhibition(exhibition_public_id):
    """
    編輯展覽（對外使用 public_id）
    GET：顯示編輯表單
    POST：處理表單並更新展覽
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 再次確認權限（裝飾器已經檢查過，這裡是雙重確認）
    if not current_user.can_manage_exhibition(exhibition):
        abort(403, "您沒有權限編輯此展覽")
    
    if request.method == "POST":
        # 取得表單資料
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        start_date_str = request.form.get("start_date", "").strip()
        end_date_str = request.form.get("end_date", "").strip()
        is_published = request.form.get("is_published") == "on"
        
        # 驗證必填欄位
        if not title:
            flash("請輸入展覽標題", "error")
            return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
        
        # 處理日期
        start_date = None
        end_date = None
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            except ValueError:
                flash("開始日期格式不正確", "error")
                return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
        
        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            except ValueError:
                flash("結束日期格式不正確", "error")
                return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
        
        # 處理封面圖片與樓層目錄（可選）
        cover_file = request.files.get("cover_image")
        if cover_file and cover_file.filename:
            filename = secure_filename(cover_file.filename)
            ext = Path(filename).suffix.lower()
            
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                flash("封面圖片格式不支援，請使用 JPG、PNG 或 WEBP", "error")
                return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
            
            # 使用現有展覽目錄或創建新目錄
            if exhibition.cover_image:
                # 如果已有封面圖片，使用相同的目錄
                existing_cover_path = BASE_DIR / exhibition.cover_image
                cover_dir = existing_cover_path.parent
            else:
                # 創建新的展覽目錄
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]
                cover_dir = EXHIBITION_DIR / safe_title
                cover_dir.mkdir(parents=True, exist_ok=True)
            
            cover_filename = f"cover{ext}"
            cover_image_path = cover_dir / cover_filename
            cover_file.save(cover_image_path)
            exhibition.cover_image = str(cover_image_path.relative_to(BASE_DIR))
        
        # 更新展覽記錄
        try:
            exhibition.title = title
            exhibition.description = description
            exhibition.start_date = start_date
            exhibition.end_date = end_date
            exhibition.is_published = is_published
            exhibition.updated_at = datetime.now()
            
            db.session.commit()

            # 更新 / 新增第一層樓層 F001（若有提供樓層資訊）
            floor_file = request.files.get("floor_image_f001")
            width_str = request.form.get("floor_width_f001", "").strip()
            height_str = request.form.get("floor_height_f001", "").strip()
            grid_str = request.form.get("floor_grid_f001", "").strip()

            # 只改 grid_size 也應該要能更新（不一定會重填寬高或重傳平面圖）
            has_floor_input = (floor_file and floor_file.filename) or (width_str and height_str) or bool(grid_str)
            if has_floor_input:
                try:
                    width_m = float(width_str) if width_str else None
                    height_m = float(height_str) if height_str else None
                    grid_size = float(grid_str) if grid_str else None
                except ValueError:
                    width_m = height_m = None
                    grid_size = None

                # 找出或建立 F001
                floor = next((f for f in exhibition.floors if f.floor_code == "F001"), None)

                safe_title = "".join(c for c in exhibition.title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]
                floor_dir = EXHIBITION_DIR / safe_title
                floor_dir.mkdir(parents=True, exist_ok=True)

                def _save_floor_image(target_floor):
                    nonlocal floor_file
                    if floor_file and floor_file.filename:
                        _filename = secure_filename(floor_file.filename)
                        _ext = Path(_filename).suffix.lower()
                        if _ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                            raise ValueError("樓層平面圖格式不支援，請使用 JPG、PNG 或 WEBP")
                        floor_filename = f"F001_floor{_ext}"
                        floor_path = floor_dir / floor_filename
                        floor_file.save(floor_path)
                        target_floor.image_path = str(floor_path.relative_to(BASE_DIR))

                try:
                    if not floor and width_m and height_m:
                        # 建立新樓層
                        floor = ExhibitionFloor(
                            exhibition_id=exhibition.id,
                            floor_code="F001",
                            image_path="",
                            width_meters=width_m,
                            height_meters=height_m,
                            grid_size=grid_size if grid_size and grid_size > 0 else 1.0,
                            created_at=datetime.now(),
                        )
                        db.session.add(floor)
                        db.session.flush()
                        _save_floor_image(floor)
                        _generate_cells_for_floor(floor)
                        
                        # 處理圈選區域（如果有）
                        selection_polygon_str = request.form.get("selection_polygon", "").strip()
                        if selection_polygon_str:
                            try:
                                polygon_data = json.loads(selection_polygon_str)
                                if polygon_data and len(polygon_data) >= 3:
                                    _apply_selection_polygon(floor, polygon_data)
                            except (json.JSONDecodeError, ValueError) as e:
                                # 如果解析失敗，忽略錯誤，不影響展覽更新
                                pass
                        
                        db.session.commit()
                    elif not floor and (grid_size and grid_size > 0) and not (width_m and height_m):
                        # 只有 grid_size 但沒有既有樓層也沒有寬高，無法建立/更新
                        flash("只修改網格大小前，請先填寫樓層的寬度與高度（或先建立 F001 樓層）", "error")
                        return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
                    elif floor:
                        updated = False
                        # 只有當使用者真的修改數值時，才更新並重建網格
                        if width_m is not None and height_m is not None:
                            if (floor.width_meters != width_m) or (floor.height_meters != height_m):
                                floor.width_meters = width_m
                                floor.height_meters = height_m
                                updated = True
                        if grid_size is not None and grid_size > 0:
                            if floor.grid_size != grid_size:
                                floor.grid_size = grid_size
                                updated = True
                        if floor_file and floor_file.filename:
                            _save_floor_image(floor)
                            updated = True

                        if updated:
                            db.session.flush()
                            _generate_cells_for_floor(floor)
                            db.session.commit()
                except ValueError as ve:
                    db.session.rollback()
                    flash(str(ve), "error")
                    return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)

            # 不論是否有修改樓層尺寸，只要有圈選資料，就依指定樓層套用
            selection_floor_code = request.form.get("selection_floor_code", "").strip()
            selection_polygon_str = request.form.get("selection_polygon", "").strip()
            if selection_polygon_str:
                try:
                    polygon_data = json.loads(selection_polygon_str)
                    if polygon_data:
                        target_floor = None
                        if selection_floor_code:
                            target_floor = next((f for f in exhibition.floors if f.floor_code == selection_floor_code), None)
                        if not target_floor:
                            # 向後相容：沒選就預設套用到 F001
                            target_floor = next((f for f in exhibition.floors if f.floor_code == "F001"), None)
                        if target_floor:
                            _apply_selection_polygon(target_floor, polygon_data)
                            db.session.commit()
                except (json.JSONDecodeError, ValueError):
                    # 圈選資料不合法時忽略，不影響展覽基本資訊更新
                    pass

            flash("展覽更新成功！", "success")
            return redirect(url_for("admin.exhibitions_list"))
        
        except Exception as e:
            db.session.rollback()
            flash(f"更新展覽失敗：{str(e)}", "error")
            return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
    
    # GET 請求：顯示編輯表單
    return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)


@admin_bp.route("/exhibitions/<exhibition_public_id>/delete", methods=["POST"])
@can_manage_exhibition('exhibition_public_id')
def delete_exhibition(exhibition_public_id):
    """
    刪除展覽（對外使用 public_id）
    只有展覽創建者或超級管理員可以刪除
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 再次確認權限
    if not current_user.can_manage_exhibition(exhibition):
        abort(403, "您沒有權限刪除此展覽")
    
    try:
        errors = []
        
        # 1. 刪除展覽照片的實體檔案（photo_path、thumbnail_path）
        for photo in list(exhibition.photos):
            if photo.photo_path:
                p = Path(photo.photo_path)
                if not p.is_absolute():
                    p = BASE_DIR / p
                if p.exists():
                    try:
                        p.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除照片檔案 {p.name}: {e}")
            if photo.thumbnail_path and photo.thumbnail_path != photo.photo_path:
                tp = Path(photo.thumbnail_path)
                if not tp.is_absolute():
                    tp = BASE_DIR / tp
                if tp.exists():
                    try:
                        tp.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除縮圖 {tp.name}: {e}")
        
        # 2. 刪除該展覽下所有 Media 的實體檔案（uploads、outputs、previews、metadata）
        for media in list(exhibition.media_files):
            media_id = media.media_id
            if media.upload_path:
                up = Path(media.upload_path)
                if not up.is_absolute():
                    up = BASE_DIR / up
                if up.exists():
                    try:
                        up.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除上傳檔案 {up.name}: {e}")
            if media.output_path:
                op = Path(media.output_path)
                if not op.is_absolute():
                    op = BASE_DIR / op
                if op.exists():
                    try:
                        op.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除處理檔案 {op.name}: {e}")
            for preview_file in PREVIEW_DIR.glob(f"{media_id}_preview.*"):
                if preview_file.exists():
                    try:
                        preview_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除預覽圖 {preview_file.name}: {e}")
            for crop_file in PREVIEW_DIR.glob(f"{media_id}_face_*.jpg"):
                if crop_file.exists():
                    try:
                        crop_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除人臉截圖 {crop_file.name}: {e}")
            faces_json = METADATA_DIR / f"{media_id}_faces.json"
            if faces_json.exists():
                try:
                    faces_json.unlink()
                except Exception as e:
                    errors.append(f"無法刪除人臉資料: {e}")
            for crop_file in METADATA_DIR.glob(f"{media_id}_face_*.jpg"):
                if crop_file.exists():
                    try:
                        crop_file.unlink()
                    except Exception as e:
                        errors.append(f"無法刪除人臉截圖 {crop_file.name}: {e}")
            db.session.delete(media)
        
        # 3. 刪除展覽（cascade 會一併刪除 ExhibitionPhoto）
        db.session.delete(exhibition)
        db.session.commit()
        
        if errors:
            flash("展覽已刪除，但部分檔案刪除失敗: " + "; ".join(errors[:3]) + ("..." if len(errors) > 3 else ""), "warning")
        else:
            flash("展覽已刪除", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"刪除展覽失敗：{str(e)}", "error")
    
    return redirect(url_for("admin.exhibitions_list"))


# ==================== 樓層與區域管理 ====================

@admin_bp.route("/exhibitions/<exhibition_public_id>/floors")
@can_manage_exhibition('exhibition_public_id')
def floors_management(exhibition_public_id):
    """
    樓層管理頁面（對外使用 public_id）
    顯示展覽的所有樓層列表
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 再次確認權限
    if not current_user.can_manage_exhibition(exhibition):
        abort(403, _("您沒有權限管理此展覽"))
    
    floors = list(exhibition.floors)
    floors.sort(key=lambda f: f.floor_code)  # 依 F001, F002... 排序
    
    return render_template("admin/floors_management.html", exhibition=exhibition, floors=floors)


@admin_bp.route("/exhibitions/<exhibition_public_id>/floors/create", methods=["GET", "POST"])
@can_manage_exhibition('exhibition_public_id')
def create_floor(exhibition_public_id):
    """
    新增樓層（對外使用 public_id）
    GET：顯示新增樓層表單
    POST：處理表單並新增樓層、產生 cells
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()

    # 再次確認權限
    if not current_user.can_manage_exhibition(exhibition):
        abort(403, _("您沒有權限管理此展覽"))

    suggested_floor_code = _suggest_next_floor_code(exhibition)

    if request.method == "POST":
        floor_code = (request.form.get("floor_code") or "").strip().upper() or suggested_floor_code
        width_str = (request.form.get("width_meters") or "").strip()
        height_str = (request.form.get("height_meters") or "").strip()
        grid_str = (request.form.get("grid_size") or "").strip()
        floor_file = request.files.get("floor_image")

        if not re.match(r"^F\d{3}$", floor_code):
            flash(_("樓層代碼格式不正確（例如 F001）"), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

        exists = ExhibitionFloor.query.filter_by(exhibition_id=exhibition.id, floor_code=floor_code).first()
        if exists:
            flash(_("此樓層代碼已存在"), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

        if not floor_file or not floor_file.filename:
            flash(_("請上傳樓層平面圖"), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

        try:
            width_m = float(width_str)
            height_m = float(height_str)
            grid_size = float(grid_str) if grid_str else 1.0
        except ValueError:
            flash(_("請輸入正確的尺寸（公尺）"), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

        if width_m <= 0 or height_m <= 0:
            flash(_("寬度與高度必須大於 0"), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

        if grid_size <= 0:
            grid_size = 1.0

        # 儲存樓層圖片
        filename = secure_filename(floor_file.filename)
        ext = Path(filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            flash(_("樓層平面圖格式不支援，請使用 JPG、PNG 或 WEBP"), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

        exhibition_dir = _get_exhibition_dir(exhibition)
        exhibition_dir.mkdir(parents=True, exist_ok=True)
        floor_filename = f"{floor_code}_floor{ext}"
        floor_path = exhibition_dir / floor_filename
        floor_file.save(floor_path)
        floor_rel = str(floor_path.relative_to(BASE_DIR))

        try:
            floor = ExhibitionFloor(
                exhibition_id=exhibition.id,
                floor_code=floor_code,
                image_path=floor_rel,
                width_meters=width_m,
                height_meters=height_m,
                grid_size=grid_size,
                created_at=datetime.now(),
            )
            db.session.add(floor)
            db.session.flush()
            _generate_cells_for_floor(floor)
            db.session.commit()
            flash(_("樓層新增成功"), "success")
            return redirect(url_for("admin.floors_management", exhibition_public_id=exhibition.public_id))
        except Exception as e:
            db.session.rollback()
            flash((_("新增樓層失敗：%(error)s") % {"error": str(e)}), "error")
            return render_template(
                "admin/floor_form.html",
                mode="create",
                exhibition=exhibition,
                suggested_floor_code=suggested_floor_code,
                form_floor_code=floor_code,
                form_width=width_str,
                form_height=height_str,
                form_grid=grid_str,
            )

    # GET
    return render_template(
        "admin/floor_form.html",
        mode="create",
        exhibition=exhibition,
        suggested_floor_code=suggested_floor_code,
        form_floor_code=suggested_floor_code,
        form_width="",
        form_height="",
        form_grid="1",
    )


@admin_bp.route("/exhibitions/<exhibition_public_id>/floors/<floor_code>/cells", methods=["GET", "POST"])
@can_manage_exhibition('exhibition_public_id')
def cells_management(exhibition_public_id, floor_code):
    """
    區域管理頁面（對外使用 public_id）
    GET：顯示該樓層的所有區域列表
    POST：批量更新區域名稱與啟用狀態
    """
    exhibition = Exhibition.query.filter_by(public_id=exhibition_public_id).first_or_404()
    
    # 再次確認權限
    if not current_user.can_manage_exhibition(exhibition):
        abort(403, "您沒有權限管理此展覽")
    
    floor = next((f for f in exhibition.floors if f.floor_code == floor_code), None)
    if not floor:
        abort(404, _("找不到該樓層"))
    
    if request.method == "POST":
        # 合併區操作：建立 / 更新名稱 / 解除合併
        action = (request.form.get("merge_action") or "").strip()
        if action == "create":
            merge_name = (request.form.get("merge_region_name") or "").strip()
            merge_cell_ids = request.form.getlist("merge_cell_ids")
            if merge_name and merge_cell_ids:
                try:
                    region = ExhibitionMergedRegion(
                        floor_id=floor.id,
                        name=merge_name,
                        display_order=len(floor.merged_regions) if floor.merged_regions else 0,
                    )
                    db.session.add(region)
                    db.session.flush()
                    for cid_str in merge_cell_ids:
                        try:
                            cid = int(cid_str)
                            cell = ExhibitionCell.query.get(cid)
                            if cell and cell.floor_id == floor.id:
                                cell.merged_region_id = region.id
                        except (ValueError, TypeError):
                            pass
                    db.session.commit()
                    flash(_("合併區「%(name)s」已建立") % {"name": merge_name}, "success")
                except Exception as e:
                    db.session.rollback()
                    flash((_("建立合併區失敗：%(error)s") % {"error": str(e)}), "error")
            else:
                flash(_("請輸入區域名稱並至少選擇一個儲存格"), "warning")
            return redirect(url_for("admin.cells_management", exhibition_public_id=exhibition.public_id, floor_code=floor_code))
        if action == "update":
            region_id = request.form.get("merge_region_id")
            new_name = (request.form.get("merge_region_new_name") or "").strip()
            if region_id and new_name:
                try:
                    region = ExhibitionMergedRegion.query.get(int(region_id))
                    if region and region.floor_id == floor.id:
                        region.name = new_name
                        db.session.commit()
                        flash(_("合併區名稱已更新"), "success")
                    else:
                        flash(_("找不到該合併區"), "error")
                except (ValueError, TypeError) as e:
                    db.session.rollback()
                    flash((_("更新失敗：%(error)s") % {"error": str(e)}), "error")
            else:
                flash(_("請輸入新名稱"), "warning")
            return redirect(url_for("admin.cells_management", exhibition_public_id=exhibition.public_id, floor_code=floor_code))
        if action == "delete":
            region_id = request.form.get("merge_region_id")
            if region_id:
                try:
                    region = ExhibitionMergedRegion.query.get(int(region_id))
                    if region and region.floor_id == floor.id:
                        for cell in list(region.cells):
                            cell.merged_region_id = None
                        db.session.delete(region)
                        db.session.commit()
                        flash(_("合併區已解除"), "success")
                    else:
                        flash(_("找不到該合併區"), "error")
                except (ValueError, TypeError) as e:
                    db.session.rollback()
                    flash((_("解除合併失敗：%(error)s") % {"error": str(e)}), "error")
            return redirect(url_for("admin.cells_management", exhibition_public_id=exhibition.public_id, floor_code=floor_code))

        # 批量更新區域名稱與啟用狀態
        cell_updates = {}
        # 先收集所有 cell_id
        all_cell_ids = set()
        for key in request.form.keys():
            if key.startswith("cell_name_"):
                all_cell_ids.add(int(key.replace("cell_name_", "")))
        
        # 處理名稱更新
        for cell_id in all_cell_ids:
            name_key = f"cell_name_{cell_id}"
            if name_key in request.form:
                cell_updates.setdefault(cell_id, {})["name"] = request.form[name_key].strip()
        
        # 處理啟用狀態：如果 checkbox 有勾選（存在於 form），is_active=True；否則 False
        for cell_id in all_cell_ids:
            active_key = f"cell_active_{cell_id}"
            if active_key in request.form:
                cell_updates.setdefault(cell_id, {})["is_active"] = True
            else:
                # checkbox 未勾選時不會出現在 form 中，所以需要明確設為 False
                cell_updates.setdefault(cell_id, {})["is_active"] = False
        
        try:
            for cell_id, updates in cell_updates.items():
                cell = ExhibitionCell.query.get(cell_id)
                if cell and cell.floor_id == floor.id:
                    if "name" in updates:
                        # 若是「預設區域名稱」，不因語言不同而把資料寫成中英混雜
                        # - 中文模式可能是「區域 C000001」
                        # - 英文模式可能是「Area C000001」
                        normalized = updates["name"]
                        try:
                            default_zh = f"區域 {cell.cell_code}"
                            default_en = f"Area {cell.cell_code}"
                            if normalized in (default_zh, default_en):
                                normalized = default_zh
                        except Exception:
                            pass
                        cell.name = normalized
                    if "is_active" in updates:
                        cell.is_active = updates["is_active"]
            
            db.session.commit()
            flash(_("區域更新成功"), "success")
        except Exception as e:
            db.session.rollback()
            flash((_("更新失敗：%(error)s") % {"error": str(e)}), "error")
        
        return redirect(url_for("admin.cells_management", exhibition_public_id=exhibition.public_id, floor_code=floor_code))
    
    # GET：顯示區域列表與合併區
    cells = list(floor.cells)
    # 依 row, col 排序（上到下、左到右）
    cells.sort(key=lambda c: (c.row, c.col))
    
    # 統計每個區域的媒體數量
    for cell in cells:
        cell.media_count = len(cell.media_files) if hasattr(cell, 'media_files') else 0
    
    merged_regions = list(floor.merged_regions) if hasattr(floor, 'merged_regions') else []
    merged_regions.sort(key=lambda r: (r.display_order, r.id))
    for r in merged_regions:
        r.cell_codes_sorted = [c.cell_code for c in sorted(r.cells, key=lambda c: (c.row, c.col))]
    merged_regions_for_plan = [
        {"name": r.name, "cells": [{"row": c.row, "col": c.col} for c in r.cells]}
        for r in merged_regions
    ]
    
    return render_template(
        "admin/cells_management.html",
        exhibition=exhibition,
        floor=floor,
        cells=cells,
        merged_regions=merged_regions,
        merged_regions_for_plan=merged_regions_for_plan,
    )


# ==================== 用戶角色管理（僅超級管理員） ====================

@admin_bp.route("/users")
@super_admin_required
def users_list():
    """
    用戶列表頁面
    只有超級管理員可以訪問
    """
    users = User.query.order_by(User.id.asc()).all()
    return render_template("admin/users_list.html", users=users)


@admin_bp.route("/users/<user_public_id>/set_role", methods=["POST"])
@super_admin_required
def set_user_role(user_public_id):
    """
    設置用戶角色（對外使用 public_id）
    只有超級管理員可以設置其他用戶的角色
    """
    user = User.query.filter_by(public_id=user_public_id).first_or_404()
    
    # 不能修改自己的角色
    if user.id == current_user.id:
        flash("您不能修改自己的角色", "error")
        return redirect(url_for("admin.users_list"))
    
    new_role = request.form.get("role", "").strip()
    
    # 驗證角色值
    valid_roles = [User.ROLE_SUPER_ADMIN, User.ROLE_ADMIN, User.ROLE_USER]
    if new_role not in valid_roles:
        flash("無效的角色", "error")
        return redirect(url_for("admin.users_list"))
    
    try:
        user.role = new_role
        # 同步更新 is_super_admin 欄位
        user.is_super_admin = (new_role == User.ROLE_SUPER_ADMIN)
        
        db.session.commit()
        
        flash(f"已將 {user.email} 的角色設置為 {new_role}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"設置角色失敗：{str(e)}", "error")
    
    return redirect(url_for("admin.users_list"))


@admin_bp.route("/users/<user_public_id>/toggle_active", methods=["POST"])
@super_admin_required
def toggle_user_active(user_public_id):
    """
    啟用/停用用戶帳號（對外使用 public_id）
    只有超級管理員可以操作
    """
    user = User.query.filter_by(public_id=user_public_id).first_or_404()
    
    # 不能停用自己的帳號
    if user.id == current_user.id:
        flash("您不能停用自己的帳號", "error")
        return redirect(url_for("admin.users_list"))
    
    try:
        user.is_active = not user.is_active
        db.session.commit()
        
        status = "啟用" if user.is_active else "停用"
        flash(f"已{status}用戶 {user.email} 的帳號", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"操作失敗：{str(e)}", "error")
    
    return redirect(url_for("admin.users_list"))
