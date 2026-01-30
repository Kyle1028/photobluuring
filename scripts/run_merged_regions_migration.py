"""
執行合併區資料庫遷移：新增 exhibition_merged_regions 表與 exhibition_cells.merged_region_id 欄位。
請先設定 DATABASE_URL 環境變數（或執行 start_with_mysql.bat 後另開終端執行此腳本）。
"""
import os
import sys
from pathlib import Path

# 專案根目錄
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 若未設定 DATABASE_URL，使用與 start_with_mysql.bat 相同的預設值
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "mysql+pymysql://root:1028@localhost:3306/photobluuring"
    print("使用預設 DATABASE_URL（可設定環境變數覆蓋）")

def main():
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("請先安裝: pip install sqlalchemy pymysql")
        return 1

    engine = create_engine(DATABASE_URL)
    migration_file = ROOT / "docs" / "migrate_add_merged_regions.sql"
    sql = migration_file.read_text(encoding="utf-8")

    # 依序執行每個語句（以分號分隔，略過註解與空行）
    statements = []
    for line in sql.split("\n"):
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        statements.append(line)
    full_sql = " ".join(statements)
    # 簡單依 ; 切分（不處理字串內的 ;）
    for stmt in full_sql.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            with engine.connect() as conn:
                conn.execute(text(stmt))
                conn.commit()
            print("OK:", stmt[:60].replace("\n", " ") + ("..." if len(stmt) > 60 else ""))
        except Exception as e:
            # 若欄位/索引/外鍵已存在則略過
            msg = str(e).lower()
            if "duplicate column" in msg or "already exists" in msg or "1060" in str(e) or "1061" in str(e):
                print("略過（已存在）:", stmt[:50], "...")
            else:
                print("錯誤:", e)
                print("語句:", stmt[:200])
                return 1

    print("\n遷移完成。請重啟 Flask 應用。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
