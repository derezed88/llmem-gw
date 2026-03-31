-- Migration 012: Coordinates cache for geocoding results
-- NOTE: This table is GLOBAL (mymcp only) — shared across all databases.
-- Do NOT use {prefix} — coordinates are location facts, not per-persona.

CREATE TABLE IF NOT EXISTS coordinates (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    place_name  VARCHAR(500) NOT NULL,
    latitude    DECIMAL(10, 7) NOT NULL,
    longitude   DECIMAL(10, 7) NOT NULL,
    formatted_address VARCHAR(500) DEFAULT NULL,
    place_id    VARCHAR(300) DEFAULT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_place_name (place_name(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
