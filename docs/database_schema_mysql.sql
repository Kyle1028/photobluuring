-- ============================================================
-- photobluuring 資料庫架構（MySQL）
-- 對應 core/models.py：users, media, exhibitions, exhibition_photos
-- ============================================================

-- 若需重建，可先 DROP 再執行（依賴順序：exhibition_photos -> media -> exhibitions -> users）
-- CREATE DATABASE IF NOT EXISTS photobluuring CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- USE photobluuring;

-- ------------------------------------------------------------
-- 1. users 使用者
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    public_id        VARCHAR(36) NULL UNIQUE,
    email           VARCHAR(120) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    username        VARCHAR(80) NULL,
    created_at      DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    last_login      DATETIME NULL,
    is_active       TINYINT(1) NULL DEFAULT 1,
    role            VARCHAR(20) NOT NULL DEFAULT 'USER',
    is_super_admin  TINYINT(1) NULL DEFAULT 0,
    verified_at     DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX ix_users_public_id (public_id),
    INDEX ix_users_email (email),
    INDEX ix_users_role (role),
    INDEX ix_users_is_super_admin (is_super_admin)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 2. exhibitions 展覽
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exhibitions (
    id           INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    public_id    VARCHAR(36) NULL UNIQUE,
    title        VARCHAR(200) NOT NULL,
    description  TEXT NULL,
    cover_image  VARCHAR(500) NULL,
    start_date   DATE NULL,
    end_date     DATE NULL,
    is_published TINYINT(1) NULL DEFAULT 1,
    created_at   DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    creator_id   INT NOT NULL,
    INDEX ix_exhibitions_public_id (public_id),
    INDEX ix_exhibitions_is_published (is_published),
    INDEX ix_exhibitions_created_at (created_at),
    CONSTRAINT fk_exhibitions_creator FOREIGN KEY (creator_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 2b. exhibition_floors 展覽樓層（含平面圖與實際尺寸）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exhibition_floors (
    id            INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    exhibition_id INT NOT NULL,
    floor_code    VARCHAR(10) NOT NULL,         -- F001, F002...
    image_path    VARCHAR(500) NOT NULL,        -- 樓層平面圖路徑（相對專案根目錄）
    width_meters  DOUBLE NOT NULL,              -- 實際寬度（公尺）
    height_meters DOUBLE NOT NULL,              -- 實際高度（公尺）
    grid_size     DOUBLE NULL DEFAULT 1.0,      -- 網格大小（公尺），預設 1m
    created_at    DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX ix_exhibition_floors_exhibition_id (exhibition_id),
    CONSTRAINT uq_exhibition_floor_code UNIQUE (exhibition_id, floor_code),
    CONSTRAINT fk_exhibition_floors_exhibition FOREIGN KEY (exhibition_id) REFERENCES exhibitions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 2b2. exhibition_merged_regions 展覽樓層合併區（多格合併為一命名區塊）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exhibition_merged_regions (
    id           INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    floor_id     INT NOT NULL,
    name         VARCHAR(200) NOT NULL,      -- 合併區顯示名稱（如「主展區」）
    display_order INT NOT NULL DEFAULT 0,
    created_at   DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX ix_exhibition_merged_regions_floor_id (floor_id),
    CONSTRAINT fk_exhibition_merged_regions_floor FOREIGN KEY (floor_id) REFERENCES exhibition_floors(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 2c. exhibition_cells 展覽樓層區域（Cell，C000001 起，每層重置）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exhibition_cells (
    id                INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    floor_id          INT NOT NULL,
    cell_code         VARCHAR(10) NOT NULL,         -- C000001...
    row               INT NOT NULL,                 -- 內部 row（上到下）
    col               INT NOT NULL,                 -- 內部 col（左到右）
    name              VARCHAR(200) NULL,
    is_active         TINYINT(1) NULL DEFAULT 1,
    merged_region_id  INT NULL,                     -- 所屬合併區（可選）
    created_at        DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX ix_exhibition_cells_floor_id (floor_id),
    INDEX ix_exhibition_cells_merged_region_id (merged_region_id),
    CONSTRAINT uq_floor_cell_code UNIQUE (floor_id, cell_code),
    CONSTRAINT uq_floor_row_col UNIQUE (floor_id, row, col),
    CONSTRAINT fk_exhibition_cells_floor FOREIGN KEY (floor_id) REFERENCES exhibition_floors(id),
    CONSTRAINT fk_exhibition_cells_merged_region FOREIGN KEY (merged_region_id) REFERENCES exhibition_merged_regions(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 3. media 媒體檔案（照片/影片）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS media (
    id            INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    media_id      VARCHAR(50) NOT NULL UNIQUE,
    original_filename VARCHAR(255) NULL,
    file_type     VARCHAR(10) NOT NULL,
    upload_path   VARCHAR(500) NULL,
    output_path   VARCHAR(500) NULL,
    process_mode  VARCHAR(20) NULL,
    face_count    INT NULL DEFAULT 0,
    status        VARCHAR(20) NULL DEFAULT 'uploaded',
    created_at    DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at  DATETIME NULL,
    user_id       INT NULL,
    exhibition_id INT NULL,
    INDEX ix_media_media_id (media_id),
    INDEX ix_media_user_id (user_id),
    INDEX ix_media_exhibition_id (exhibition_id),
    CONSTRAINT fk_media_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT fk_media_exhibition FOREIGN KEY (exhibition_id) REFERENCES exhibitions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 3b. media_cells 媒體與展覽區域關聯（多對多）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS media_cells (
    media_id INT NOT NULL,
    cell_id  INT NOT NULL,
    PRIMARY KEY (media_id, cell_id),
    CONSTRAINT fk_media_cells_media FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE,
    CONSTRAINT fk_media_cells_cell FOREIGN KEY (cell_id) REFERENCES exhibition_cells(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 4. exhibition_photos 展覽照片（展覽內的顯示項目）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exhibition_photos (
    id             INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    exhibition_id  INT NOT NULL,
    photo_path     VARCHAR(500) NOT NULL,
    thumbnail_path VARCHAR(500) NULL,
    title          VARCHAR(200) NULL,
    description    TEXT NULL,
    display_order  INT NULL DEFAULT 0,
    created_at     DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX ix_exhibition_photos_exhibition_id (exhibition_id),
    CONSTRAINT fk_exhibition_photos_exhibition FOREIGN KEY (exhibition_id) REFERENCES exhibitions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 關聯簡表（畫 ER 圖用）
-- ============================================================
-- users (1) ----< media         : user_id -> users.id
-- users (1) ----< exhibitions   : creator_id -> users.id
-- exhibitions (1) ----< media    : exhibition_id -> exhibitions.id
-- exhibitions (1) ----< exhibition_photos : exhibition_id -> exhibitions.id
-- exhibitions (1) ----< exhibition_floors  : exhibition_id -> exhibitions.id
-- exhibition_floors (1) ----< exhibition_merged_regions : floor_id -> exhibition_floors.id
-- exhibition_floors (1) ----< exhibition_cells : floor_id -> exhibition_floors.id
-- exhibition_merged_regions (1) ----< exhibition_cells : merged_region_id -> exhibition_merged_regions.id
-- media (M) ----< media_cells >---- (M) exhibition_cells : media_id/cell_id

-- ============================================================
-- 既有資料庫升級：合併區功能（若已存在 exhibition_cells 表）
-- ============================================================
-- 1. 建立合併區表（若尚未建立）：
--    CREATE TABLE IF NOT EXISTS exhibition_merged_regions (...); 見上方 2b2。
-- 2. 為 exhibition_cells 新增欄位（若尚未有 merged_region_id）：
--    ALTER TABLE exhibition_cells ADD COLUMN merged_region_id INT NULL AFTER is_active;
--    ALTER TABLE exhibition_cells ADD INDEX ix_exhibition_cells_merged_region_id (merged_region_id);
--    ALTER TABLE exhibition_cells ADD CONSTRAINT fk_exhibition_cells_merged_region
--      FOREIGN KEY (merged_region_id) REFERENCES exhibition_merged_regions(id) ON DELETE SET NULL;
