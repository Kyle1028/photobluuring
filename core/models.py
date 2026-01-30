"""
資料庫模型定義檔案
定義了使用者(User)、媒體檔案(Media)、展覽(Exhibition)和展覽照片(ExhibitionPhoto)資料表
"""

import secrets
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

# 建立資料庫物件，用於操作資料庫
db = SQLAlchemy()


def _digits(n: int, k: int = 10) -> str:
    """產生 n 個 0–9 的隨機數字。"""
    return "".join(str(secrets.randbelow(10)) for _ in range(n))


def _public_id_user(seq_id: int) -> str:
    """User 對外識別碼：6 + 11 位數字序號 + 8 碼隨機數字，共 20 字元。例：600000000001XXXXXXXX"""
    return "6" + str(seq_id).zfill(11) + _digits(8)


def _public_id_exhibition(seq_id: int) -> str:
    """展覽對外識別碼：7 + 9 位數字序號 + 6 碼隨機數字，共 16 字元。例：700000000001XXXXXXXX"""
    return "7" + str(seq_id).zfill(9) + _digits(6)


def _media_id_from_seq(seq_id: int) -> str:
    """影像檔對外識別碼：8 + 15 位數字序號 + 4 碼隨機數字，共 20 字元。例：8000000000000001XXXX"""
    return "8" + str(seq_id).zfill(15) + _digits(4)


def _refresh_media_id_suffix(media_id: str) -> str:
    """編修時只重算最後 4 碼隨機數字（須為 8+15+4 格式）。"""
    if len(media_id) == 20 and media_id[0] == "8":
        return media_id[:16] + _digits(4)
    return media_id


def _default_public_id():
    """未經「先 commit 再填 public_id」時為 None；新建請用 _public_id_user(id) / _public_id_exhibition(id)。"""
    return None


class User(UserMixin, db.Model):
    """
    使用者資料表
    儲存註冊使用者的帳號資訊
    """
    __tablename__ = "users"  # 資料表名稱
    
    # 角色常數
    ROLE_SUPER_ADMIN = "SUPER_ADMIN"
    ROLE_ADMIN = "ADMIN"
    ROLE_USER = "USER"
    
    # 欄位定義
    id = db.Column(db.Integer, primary_key=True)  # 使用者 ID（主鍵，自動遞增，僅內部使用）
    public_id = db.Column(db.String(36), unique=True, nullable=True, index=True, default=_default_public_id)  # 對外識別碼（UUID），用於 URL，不可猜
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)  # 電子郵件（唯一，不可為空）
    password_hash = db.Column(db.String(255), nullable=False)  # 加密後的密碼
    username = db.Column(db.String(80), nullable=True)  # 使用者名稱（可選）
    created_at = db.Column(db.DateTime, default=datetime.now)  # 註冊時間
    last_login = db.Column(db.DateTime)  # 最後登入時間
    is_active = db.Column(db.Boolean, default=True)  # 帳號是否啟用（預設：啟用）
    role = db.Column(db.String(20), nullable=False, default=ROLE_USER, index=True)  # 角色（SUPER_ADMIN, ADMIN, USER）
    is_super_admin = db.Column(db.Boolean, default=False, index=True)  # 是否為超級管理員（快速查詢用）
    verified_at = db.Column(db.DateTime, default=datetime.now)  # 帳號驗證時間
    
    def set_password(self, password):
        """
        設定密碼（會自動加密）
        參數：password - 使用者輸入的明文密碼
        """
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """
        驗證密碼是否正確
        參數：password - 使用者輸入的明文密碼
        回傳：True（正確）或 False（錯誤）
        """
        return check_password_hash(self.password_hash, password)
    
    def is_super_admin_role(self):
        """
        檢查是否為超級管理員
        回傳：True（是超級管理員）或 False（不是）
        """
        return self.is_super_admin or self.role == self.ROLE_SUPER_ADMIN
    
    def is_admin_role(self):
        """
        檢查是否為管理員（包含超級管理員）
        回傳：True（是管理員）或 False（不是）
        """
        return self.is_super_admin_role() or self.role == self.ROLE_ADMIN
    
    def can_manage_exhibition(self, exhibition):
        """
        檢查是否可以管理指定的展覽
        參數：exhibition - Exhibition 物件
        回傳：True（可以管理）或 False（不可以）
        
        規則：
        - 超級管理員可以管理所有展覽
        - 展覽創辦人可以管理自己創建的展覽（不論角色）
        - 一般管理員只能管理自己創建的展覽
        """
        if not exhibition:
            return False
        
        # 超級管理員可以管理所有展覽
        if self.is_super_admin_role():
            return True
        
        # 展覽創辦人可以管理自己創建的展覽（不論角色）
        if exhibition.creator_id == self.id:
            return True
        
        # 一般管理員只能管理自己創建的展覽（已在上面處理，此處保留以備未來擴展）
        if self.is_admin_role():
            return False
        
        # 其他情況無法管理
        return False
    
    def can_set_user_role(self):
        """
        檢查是否可以設置其他用戶的角色
        回傳：True（可以）或 False（不可以）
        
        只有超級管理員可以設置其他用戶的角色
        """
        return self.is_super_admin_role()
    
    def __repr__(self):
        """物件的字串表示（用於除錯）"""
        return f"<User {self.email}>"


media_cells = db.Table(
    "media_cells",
    db.Column("media_id", db.Integer, db.ForeignKey("media.id"), primary_key=True),
    db.Column("cell_id", db.Integer, db.ForeignKey("exhibition_cells.id"), primary_key=True),
)


class Media(db.Model):
    """
    媒體檔案資料表
    儲存使用者上傳的照片/影片資訊
    """
    __tablename__ = "media"  # 資料表名稱
    
    # 欄位定義
    id = db.Column(db.Integer, primary_key=True)  # 媒體檔案 ID（主鍵，自動遞增）
    media_id = db.Column(db.String(50), unique=True, nullable=False, index=True)  # 媒體檔案的唯一識別碼
    original_filename = db.Column(db.String(255))  # 原始檔案名稱
    file_type = db.Column(db.String(10), nullable=False)  # 檔案類型（image 或 video）
    upload_path = db.Column(db.String(500))  # 上傳檔案的儲存路徑
    output_path = db.Column(db.String(500))  # 處理後檔案的儲存路徑
    process_mode = db.Column(db.String(20))  # 處理模式（mosaic：馬賽克 / eyes：遮眼 / replace：替換）
    face_count = db.Column(db.Integer, default=0)  # 偵測到的人臉數量
    status = db.Column(db.String(20), default="uploaded")  # 檔案狀態（uploaded：已上傳 / processed：已處理）
    created_at = db.Column(db.DateTime, default=datetime.now)  # 上傳時間
    processed_at = db.Column(db.DateTime)  # 處理完成時間
    
    # 關聯欄位：這個媒體檔案屬於哪個使用者（加 index 以加速 filter_by(user_id)）
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    
    # 關聯欄位：這個媒體檔案屬於哪個展覽（加 index 以加速 filter_by(exhibition_id)）
    exhibition_id = db.Column(db.Integer, db.ForeignKey("exhibitions.id"), nullable=True, index=True)
    
    # 關聯：上傳者（多對一）
    user = db.relationship("User", backref="media_files")

    # 關聯：所屬展覽區域（多對多，一個媒體可以掛在多個 Cell 上）
    cells = db.relationship(
        "ExhibitionCell",
        secondary=media_cells,
        backref="media_files",
    )
    
    def __repr__(self):
        """物件的字串表示（用於除錯）"""
        return f"<Media {self.media_id}>"


class Exhibition(db.Model):
    """
    展覽資料表
    儲存展覽的基本資訊
    """
    __tablename__ = "exhibitions"
    __table_args__ = {'extend_existing': True}  # 允許擴展現有表定義
    
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(36), unique=True, nullable=True, index=True, default=_default_public_id)  # 對外識別碼（UUID），用於 URL，不可猜
    title = db.Column(db.String(200), nullable=False)  # 展覽標題
    description = db.Column(db.Text)  # 展覽描述
    cover_image = db.Column(db.String(500))  # 封面圖片路徑
    start_date = db.Column(db.Date)  # 開始日期
    end_date = db.Column(db.Date)  # 結束日期
    is_published = db.Column(db.Boolean, default=True, index=True)  # 是否公開（首頁/列表常用 filter）
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)  # 建立時間（常用 order_by）
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)  # 更新時間
    
    # 關聯欄位：展覽的建立者
    creator_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    
    # 關聯：展覽的照片（一對多）
    photos = db.relationship("ExhibitionPhoto", back_populates="exhibition", cascade="all, delete-orphan", order_by="ExhibitionPhoto.display_order")
    
    # 關聯：展覽的媒體檔案（一對多）
    media_files = db.relationship("Media", backref="exhibition", lazy="dynamic")

    # 關聯：展覽的樓層（一對多，例如 F001、F002...）
    floors = db.relationship(
        "ExhibitionFloor",
        back_populates="exhibition",
        cascade="all, delete-orphan",
        lazy="select",
    )
    
    def __repr__(self):
        return f"<Exhibition {self.title}>"


class ExhibitionMergedRegion(db.Model):
    """
    展覽樓層上的合併區（多個 Cell 合併為一個命名區塊）
    展覽顯示時會以區塊為單位自成一個網格並顯示區域名稱。
    """
    __tablename__ = "exhibition_merged_regions"

    id = db.Column(db.Integer, primary_key=True)
    floor_id = db.Column(db.Integer, db.ForeignKey("exhibition_floors.id"), nullable=False, index=True)

    # 合併區顯示名稱（例如「主展區」「服務台」）
    name = db.Column(db.String(200), nullable=False)

    # 顯示順序（數字越小越前面）
    display_order = db.Column(db.Integer, default=0)

    created_at = db.Column(db.DateTime, default=datetime.now)

    floor = db.relationship("ExhibitionFloor", back_populates="merged_regions")
    cells = db.relationship("ExhibitionCell", back_populates="merged_region", foreign_keys="ExhibitionCell.merged_region_id")

    def __repr__(self):
        return f"<ExhibitionMergedRegion {self.floor_id}:{self.name}>"


class ExhibitionFloor(db.Model):
    """
    展覽樓層資料表（含平面圖與實際尺寸）
    例：F001, F002 ...
    """
    __tablename__ = "exhibition_floors"

    id = db.Column(db.Integer, primary_key=True)
    exhibition_id = db.Column(db.Integer, db.ForeignKey("exhibitions.id"), nullable=False, index=True)

    # 對外樓層代碼，例如 F001、F002（每個展覽內唯一）
    floor_code = db.Column(db.String(10), nullable=False)

    # 樓層平面圖路徑（相對於專案根目錄）
    image_path = db.Column(db.String(500), nullable=False)

    # 實際尺寸（單位：公尺）
    width_meters = db.Column(db.Float, nullable=False)
    height_meters = db.Column(db.Float, nullable=False)

    # 此樓層的網格大小（公尺），預設 1m×1m，可依樓層調整
    grid_size = db.Column(db.Float, default=1.0)

    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    exhibition = db.relationship("Exhibition", back_populates="floors")
    cells = db.relationship("ExhibitionCell", back_populates="floor", cascade="all, delete-orphan")
    merged_regions = db.relationship("ExhibitionMergedRegion", back_populates="floor", cascade="all, delete-orphan")

    __table_args__ = (
        db.UniqueConstraint("exhibition_id", "floor_code", name="uq_exhibition_floor_code"),
    )

    def __repr__(self):
        return f"<ExhibitionFloor {self.exhibition_id}:{self.floor_code}>"


class ExhibitionCell(db.Model):
    """
    展覽樓層上的區域（Cell）
    UI 對外使用 cell_code（例如 C000001），內部使用 row / col 做排序與點擊定位。
    可選屬於某個合併區（merged_region_id），展覽顯示時會依合併區分塊呈現。
    """
    __tablename__ = "exhibition_cells"

    id = db.Column(db.Integer, primary_key=True)
    floor_id = db.Column(db.Integer, db.ForeignKey("exhibition_floors.id"), nullable=False, index=True)

    # 對外區域代碼，例如 C000001（每層重置）
    cell_code = db.Column(db.String(10), nullable=False)

    # 內部用網格座標：row 從上到下、col 從左到右（從 0 起算）
    row = db.Column(db.Integer, nullable=False)
    col = db.Column(db.Integer, nullable=False)

    # 顯示名稱（可由創建人自訂），預設可為「區域 C000001」之類
    name = db.Column(db.String(200))

    # 是否為有效區域（牆壁、樓梯等可標記為 False）
    is_active = db.Column(db.Boolean, default=True)

    # 所屬合併區（可選）；同一合併區的儲存格在展覽頁會自成一個網格區塊並顯示區域名稱
    merged_region_id = db.Column(db.Integer, db.ForeignKey("exhibition_merged_regions.id"), nullable=True, index=True)

    created_at = db.Column(db.DateTime, default=datetime.now)

    floor = db.relationship("ExhibitionFloor", back_populates="cells")
    merged_region = db.relationship("ExhibitionMergedRegion", back_populates="cells", foreign_keys=[merged_region_id])

    __table_args__ = (
        db.UniqueConstraint("floor_id", "cell_code", name="uq_floor_cell_code"),
        db.UniqueConstraint("floor_id", "row", "col", name="uq_floor_row_col"),
    )

    def __repr__(self):
        return f"<ExhibitionCell {self.floor_id}:{self.cell_code} ({self.row},{self.col})>"


class ExhibitionPhoto(db.Model):
    """
    展覽照片資料表
    儲存展覽中的照片資訊
    """
    __tablename__ = "exhibition_photos"
    __table_args__ = {'extend_existing': True}  # 允許擴展現有表定義
    
    id = db.Column(db.Integer, primary_key=True)
    photo_path = db.Column(db.String(500), nullable=False)  # 照片路徑
    thumbnail_path = db.Column(db.String(500))  # 縮圖路徑
    title = db.Column(db.String(200))  # 照片標題
    description = db.Column(db.Text)  # 照片描述
    display_order = db.Column(db.Integer, default=0)  # 顯示順序
    created_at = db.Column(db.DateTime, default=datetime.now)  # 上傳時間
    
    # 關聯欄位：照片所屬的展覽（加 index 以加速 filter_by(exhibition_id)）
    exhibition_id = db.Column(db.Integer, db.ForeignKey("exhibitions.id"), nullable=False, index=True)
    
    # 關聯：展覽物件
    exhibition = db.relationship("Exhibition", back_populates="photos")
    
    def __repr__(self):
        return f"<ExhibitionPhoto {self.id}>"
