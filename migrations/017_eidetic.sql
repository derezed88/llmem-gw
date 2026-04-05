-- Migration 017: Eidetic (visual/photo) memory
-- Stores photo analysis results linked to Google Drive images.
-- Uses {prefix} so each database gets its own eidetic table.

CREATE TABLE IF NOT EXISTS `{prefix}memory_eidetic` (
    `id`             INT AUTO_INCREMENT PRIMARY KEY,
    `topic`          VARCHAR(255) NOT NULL,
    `content`        TEXT NOT NULL,
    `importance`     TINYINT NOT NULL DEFAULT 5 COMMENT '1=low 5=med 10=critical',
    `source`         ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`     VARCHAR(255) DEFAULT NULL,
    `drive_file_id`  VARCHAR(128) DEFAULT NULL,
    `task_type`      VARCHAR(32) DEFAULT 'general',
    `analysis_model` VARCHAR(64) DEFAULT 'gemini-2.5-flash',
    `memory_link`    TEXT DEFAULT NULL COMMENT 'JSON array of related memory row IDs',
    `location_lat`   DECIMAL(10, 7) DEFAULT NULL,
    `location_lon`   DECIMAL(10, 7) DEFAULT NULL,
    `created_at`     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at`     TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY `idx_topic`         (`topic`),
    KEY `idx_importance`    (`importance`),
    KEY `idx_drive_file_id` (`drive_file_id`),
    KEY `idx_created`       (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
