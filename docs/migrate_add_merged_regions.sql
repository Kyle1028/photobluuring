-- ============================================================
-- 遷移：新增合併區功能（既有資料庫請執行此檔一次）
-- 若 exhibition_merged_regions 表或 exhibition_cells.merged_region_id 已存在，請略過對應的語句。
-- ============================================================

-- 1. 建立合併區表
CREATE TABLE IF NOT EXISTS exhibition_merged_regions (
    id           INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    floor_id     INT NOT NULL,
    name         VARCHAR(200) NOT NULL,
    display_order INT NOT NULL DEFAULT 0,
    created_at   DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX ix_exhibition_merged_regions_floor_id (floor_id),
    CONSTRAINT fk_exhibition_merged_regions_floor
        FOREIGN KEY (floor_id) REFERENCES exhibition_floors(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. 在 exhibition_cells 新增欄位（若已存在會報錯，可略過）
ALTER TABLE exhibition_cells
    ADD COLUMN merged_region_id INT NULL AFTER is_active;

ALTER TABLE exhibition_cells
    ADD INDEX ix_exhibition_cells_merged_region_id (merged_region_id);

ALTER TABLE exhibition_cells
    ADD CONSTRAINT fk_exhibition_cells_merged_region
        FOREIGN KEY (merged_region_id) REFERENCES exhibition_merged_regions(id) ON DELETE SET NULL;
