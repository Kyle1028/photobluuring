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
