"""
創建展覽的腳本
將 test/展覽圖.jpg 設置為首頁展覽
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date
from shutil import copy2

# 添加專案根目錄到路徑
# BASE_DIR 應該是專案根目錄，不是 scripts 目錄
BASE_DIR = Path(__file__).resolve().parent.parent  # 上一層目錄才是專案根目錄
sys.path.insert(0, str(BASE_DIR))

# 設定 UTF-8 輸出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 檢查環境變數
if not os.environ.get("DATABASE_URL"):
    print("錯誤：缺少 DATABASE_URL 環境變數")
    print("請先設置環境變數，例如：")
    print("  set DATABASE_URL=mysql+pymysql://root:password@localhost:3306/photobluuring")
    print("\n或者如果您使用的是 SQLite，請設置：")
    print("  set DATABASE_URL=sqlite:///database/app.db")
    sys.exit(1)

# 初始化 Flask 應用
from app import app
from core.models import db, Exhibition, ExhibitionPhoto, User

def create_exhibition():
    """創建展覽記錄"""
    with app.app_context():
        # 檢查是否已經存在這個展覽
        existing = Exhibition.query.filter_by(title="台北國際航太暨國防工業展").first()
        if existing:
            print(f"展覽已存在: {existing.title} (ID: {existing.id})")
            return existing
        
        # 獲取第一個用戶作為創建者（如果沒有用戶，創建一個預設的）
        creator = User.query.first()
        if not creator:
            print("錯誤：資料庫中沒有用戶，請先註冊一個用戶")
            return None
        
        # 創建展覽目錄
        EXHIBITION_DIR = BASE_DIR / "exhibitions"
        EXHIBITION_DIR.mkdir(parents=True, exist_ok=True)
        
        # 複製圖片到展覽目錄
        source_image = BASE_DIR / "test" / "展覽圖.jpg"
        if not source_image.exists():
            print(f"錯誤：找不到圖片檔案 {source_image}")
            return None
        
        # 創建展覽專用目錄
        exhibition_dir = EXHIBITION_DIR / "tadte_2025"
        exhibition_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製圖片
        dest_image = exhibition_dir / "cover.jpg"
        copy2(source_image, dest_image)
        print(f"已複製圖片: {dest_image}")
        
        # 創建展覽記錄
        exhibition = Exhibition(
            title="台北國際航太暨國防工業展",
            description="知洋科技股份有限公司 敬邀參觀\n無人載具區 展位 I0224\n\n展示 Awareocean 台和1號 (A001) 與 創龍 (B001) 無人載具",
            cover_image=str(dest_image.relative_to(BASE_DIR)),
            start_date=date(2025, 9, 18),
            end_date=date(2025, 9, 20),
            is_published=True,
            creator_id=creator.id,
            created_at=datetime.now()
        )
        
        db.session.add(exhibition)
        db.session.flush()  # 獲取 exhibition.id
        
        # 創建展覽照片記錄
        photo = ExhibitionPhoto(
            exhibition_id=exhibition.id,
            photo_path=str(dest_image.relative_to(BASE_DIR)),
            thumbnail_path=str(dest_image.relative_to(BASE_DIR)),
            title="展覽宣傳海報",
            description="台北國際航太暨國防工業展 - 知洋科技無人載具展示",
            display_order=0,
            created_at=datetime.now()
        )
        
        db.session.add(photo)
        db.session.commit()
        
        print(f"✓ 成功創建展覽: {exhibition.title} (ID: {exhibition.id})")
        print(f"✓ 展覽照片已添加: {photo.title}")
        
        return exhibition

if __name__ == "__main__":
    try:
        create_exhibition()
        print("\n完成！請重新啟動應用程式以查看展覽。")
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
