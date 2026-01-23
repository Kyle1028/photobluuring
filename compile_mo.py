"""
手動編譯翻譯檔案的腳本
將 .po 檔案編譯成 .mo 檔案
"""
from pathlib import Path
import sys

# 設定 UTF-8 輸出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from babel.messages.pofile import read_po
    from babel.messages.mofile import write_mo
    
    # 編譯英文翻譯
    po_path = Path("translations/en/LC_MESSAGES/messages.po")
    mo_path = Path("translations/en/LC_MESSAGES/messages.mo")
    
    if po_path.exists():
        with open(po_path, 'rb') as f:
            catalog = read_po(f)
        
        with open(mo_path, 'wb') as f:
            write_mo(f, catalog)
        
        print(f"OK - Compiled: {mo_path}")
    else:
        print(f"ERROR - Not found: {po_path}")
    
    print("\nTranslation files compiled successfully! Restart the app to see the changes.")
    
except ImportError:
    print("Please install Babel first: pip install Babel")
except Exception as e:
    print(f"Compilation failed: {e}")
