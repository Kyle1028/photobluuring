"""
管理員功能模組
處理展覽管理、用戶角色管理等功能
"""

from datetime import datetime, date
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from pathlib import Path
from werkzeug.utils import secure_filename

from core.models import db, User, Exhibition, ExhibitionPhoto
from core.decorators import admin_required, super_admin_required, can_manage_exhibition

# 建立管理員藍圖，所有管理員相關的路由都以 /admin 開頭
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# 目錄設定（避免循環導入，直接在此定義）
BASE_DIR = Path(__file__).resolve().parent.parent  # 專案根目錄（core 的上一層）
EXHIBITION_DIR = BASE_DIR / "exhibitions"  # 展覽照片目錄


def init_admin(app):
    """
    初始化管理員系統
    在 Flask 應用程式啟動時呼叫此函式
    """
    app.register_blueprint(admin_bp)


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
        
        # 處理封面圖片上傳
        cover_image_path = None
        cover_file = request.files.get("cover_image")
        if cover_file and cover_file.filename:
            # 生成安全的檔案名稱
            filename = secure_filename(cover_file.filename)
            ext = Path(filename).suffix.lower()
            
            # 只允許圖片格式
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                flash("封面圖片格式不支援，請使用 JPG、PNG 或 WEBP", "error")
                return render_template("admin/exhibition_form.html", mode="create")
            
            # 創建展覽專用目錄（使用標題的簡化版本）
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')[:50]  # 限制長度
            exhibition_dir = EXHIBITION_DIR / safe_title
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
            
            flash("展覽創建成功！", "success")
            return redirect(url_for("admin.exhibitions_list"))
        
        except Exception as e:
            db.session.rollback()
            flash(f"創建展覽失敗：{str(e)}", "error")
            return render_template("admin/exhibition_form.html", mode="create")
    
    # GET 請求：顯示創建表單
    return render_template("admin/exhibition_form.html", mode="create")


@admin_bp.route("/exhibitions/<int:exhibition_id>/edit", methods=["GET", "POST"])
@can_manage_exhibition('exhibition_id')
def edit_exhibition(exhibition_id):
    """
    編輯展覽
    GET：顯示編輯表單
    POST：處理表單並更新展覽
    """
    exhibition = Exhibition.query.get_or_404(exhibition_id)
    
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
        
        # 處理封面圖片上傳（可選）
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
            
            flash("展覽更新成功！", "success")
            return redirect(url_for("admin.exhibitions_list"))
        
        except Exception as e:
            db.session.rollback()
            flash(f"更新展覽失敗：{str(e)}", "error")
            return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)
    
    # GET 請求：顯示編輯表單
    return render_template("admin/exhibition_form.html", mode="edit", exhibition=exhibition)


@admin_bp.route("/exhibitions/<int:exhibition_id>/delete", methods=["POST"])
@can_manage_exhibition('exhibition_id')
def delete_exhibition(exhibition_id):
    """
    刪除展覽
    只有展覽創建者或超級管理員可以刪除
    """
    exhibition = Exhibition.query.get_or_404(exhibition_id)
    
    # 再次確認權限
    if not current_user.can_manage_exhibition(exhibition):
        abort(403, "您沒有權限刪除此展覽")
    
    try:
        # 刪除展覽（相關的照片會因為 cascade 設定自動刪除）
        db.session.delete(exhibition)
        db.session.commit()
        
        flash("展覽已刪除", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"刪除展覽失敗：{str(e)}", "error")
    
    return redirect(url_for("admin.exhibitions_list"))


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


@admin_bp.route("/users/<int:user_id>/set_role", methods=["POST"])
@super_admin_required
def set_user_role(user_id):
    """
    設置用戶角色
    只有超級管理員可以設置其他用戶的角色
    """
    user = User.query.get_or_404(user_id)
    
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


@admin_bp.route("/users/<int:user_id>/toggle_active", methods=["POST"])
@super_admin_required
def toggle_user_active(user_id):
    """
    啟用/停用用戶帳號
    只有超級管理員可以操作
    """
    user = User.query.get_or_404(user_id)
    
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
