from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
import re


auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# 初始化 Flask-Login
login_manager = LoginManager()


def init_auth(app):
    """初始化認證系統"""
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "請先登入以訪問此頁面"
    login_manager.login_message_category = "warning"
    
    app.register_blueprint(auth_bp)


@login_manager.user_loader
def load_user(user_id):
    """載入使用者"""
    return User.query.get(int(user_id))


def validate_email(email):
    """驗證電子郵件格式"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """
    驗證密碼強度
    至少 8 個字元，包含英文和數字
    """
    if len(password) < 8:
        return False, "密碼長度至少需要 8 個字元"
    
    if not re.search(r'[A-Za-z]', password):
        return False, "密碼必須包含英文字母"
    
    if not re.search(r'\d', password):
        return False, "密碼必須包含數字"
    
    return True, "密碼符合要求"


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """註冊頁面"""
    # 如果已登入，導向首頁
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        password_confirm = request.form.get("password_confirm", "").strip()
        username = request.form.get("username", "").strip()
        
        # 驗證必填欄位
        if not email or not password or not password_confirm:
            flash("請填寫所有必填欄位", "error")
            return render_template("register.html")
        
        # 驗證電子郵件格式
        if not validate_email(email):
            flash("電子郵件格式不正確", "error")
            return render_template("register.html")
        
        # 驗證密碼
        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, "error")
            return render_template("register.html")
        
        # 確認密碼一致
        if password != password_confirm:
            flash("兩次輸入的密碼不一致", "error")
            return render_template("register.html")
        
        # 檢查電子郵件是否已註冊
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("此電子郵件已被註冊", "error")
            return render_template("register.html")
        
        try:
            # 直接建立新使用者
            new_user = User(
                email=email,
                username=username if username else email.split("@")[0]
            )
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash("註冊成功！現在可以登入了", "success")
            return redirect(url_for("auth.login"))
            
        except Exception as e:
            db.session.rollback()
            flash(f"註冊失敗：{str(e)}", "error")
            return render_template("register.html")
    
    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """登入頁面"""
    # 如果已登入，導向首頁
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        remember = request.form.get("remember") == "on"
        
        # 驗證必填欄位
        if not email or not password:
            flash("請輸入電子郵件和密碼", "error")
            return render_template("login.html")
        
        # 查詢使用者
        user = User.query.filter_by(email=email).first()
        
        # 驗證使用者和密碼
        if not user or not user.check_password(password):
            flash("電子郵件或密碼錯誤", "error")
            return render_template("login.html")
        
        # 檢查帳號是否啟用
        if not user.is_active:
            flash("您的帳號已被停用，請聯絡管理員", "error")
            return render_template("login.html")
        
 
        
        # 登入使用者
        login_user(user, remember=remember)
        
        # 更新最後登入時間
        user.last_login = datetime.now()
        db.session.commit()
        
        flash(f"歡迎回來，{user.username}！", "success")
        
        next_page = request.args.get("next")
        if next_page:
            return redirect(next_page)
        return redirect(url_for("index"))
    
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    """登出"""
    logout_user()
    flash("您已成功登出", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile")
@login_required
def profile():
    """個人資料頁面"""
    return render_template("profile.html", user=current_user)


@auth_bp.route("/update_username", methods=["POST"])
@login_required
def update_username():
    """更新使用者名稱"""
    new_username = request.form.get("new_username", "").strip()
    
    if not new_username:
        flash("使用者名稱不能為空", "error")
        return redirect(url_for("auth.profile"))
    
    if len(new_username) > 80:
        flash("使用者名稱不能超過 80 個字元", "error")
        return redirect(url_for("auth.profile"))
    
    try:
        current_user.username = new_username
        db.session.commit()
        flash("使用者名稱已更新！", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"更新失敗：{str(e)}", "error")
    
    return redirect(url_for("auth.profile"))


@auth_bp.route("/update_email", methods=["POST"])
@login_required
def update_email():
    """更新電子郵件"""
    new_email = request.form.get("new_email", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()
    
    if not new_email or not confirm_password:
        flash("請填寫所有欄位", "error")
        return redirect(url_for("auth.profile"))
    
    # 驗證密碼
    if not current_user.check_password(confirm_password):
        flash("密碼驗證失敗", "error")
        return redirect(url_for("auth.profile"))
    
    # 驗證電子郵件格式
    if not validate_email(new_email):
        flash("電子郵件格式不正確", "error")
        return redirect(url_for("auth.profile"))
    
    # 檢查電子郵件是否已被使用
    existing_user = User.query.filter_by(email=new_email).first()
    if existing_user and existing_user.id != current_user.id:
        flash("此電子郵件已被其他帳號使用", "error")
        return redirect(url_for("auth.profile"))
    
    try:
        current_user.email = new_email
        db.session.commit()
        flash("電子郵件已更新！", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"更新失敗：{str(e)}", "error")
    
    return redirect(url_for("auth.profile"))


@auth_bp.route("/change_password", methods=["POST"])
@login_required
def change_password():
    """變更密碼"""
    current_password = request.form.get("current_password", "").strip()
    new_password = request.form.get("new_password", "").strip()
    new_password_confirm = request.form.get("new_password_confirm", "").strip()
    
    if not current_password or not new_password or not new_password_confirm:
        flash("請填寫所有欄位", "error")
        return redirect(url_for("auth.profile"))
    
    # 驗證目前密碼
    if not current_user.check_password(current_password):
        flash("目前密碼錯誤", "error")
        return redirect(url_for("auth.profile"))
    
    # 驗證新密碼強度
    is_valid, message = validate_password(new_password)
    if not is_valid:
        flash(message, "error")
        return redirect(url_for("auth.profile"))
    
    # 確認新密碼一致
    if new_password != new_password_confirm:
        flash("兩次輸入的新密碼不一致", "error")
        return redirect(url_for("auth.profile"))
    
    # 檢查新密碼是否與舊密碼相同
    if current_password == new_password:
        flash("新密碼不能與目前密碼相同", "error")
        return redirect(url_for("auth.profile"))
    
    try:
        current_user.set_password(new_password)
        db.session.commit()
        flash("密碼已更新！請使用新密碼登入", "success")
        # 變更密碼後登出，要求重新登入
        logout_user()
        return redirect(url_for("auth.login"))
    except Exception as e:
        db.session.rollback()
        flash(f"更新失敗：{str(e)}", "error")
        return redirect(url_for("auth.profile"))

