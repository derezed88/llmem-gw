-- Migration 016: GPS location log
-- Stores GPS coordinates stripped from user prompts.
-- Uses {prefix} so each database gets its own location table.

CREATE TABLE IF NOT EXISTS `{prefix}location` (
    `id`         INT AUTO_INCREMENT PRIMARY KEY,
    `latitude`   DECIMAL(10, 7) NOT NULL,
    `longitude`  DECIMAL(10, 7) NOT NULL,
    `accuracy_m` FLOAT DEFAULT NULL,
    `session_id` VARCHAR(255) DEFAULT NULL,
    `created_at` DATETIME NOT NULL,
    INDEX idx_created_at (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
