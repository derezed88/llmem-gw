-- Migration: 007_temporal
-- Creates the temporal pattern cache table.
-- Stores results of temporal pattern queries (explicit user queries + inferred
-- interest queries from the reasoning timer) for fast cached retrieval.
-- HWM/LWM aging applies with higher watermarks since rows are not auto-injected.
-- Run: python migrations/apply.py 007_temporal.sql

CREATE TABLE IF NOT EXISTS `{prefix}temporal` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `source`      ENUM('explicit','inferred') NOT NULL DEFAULT 'explicit',
    `query_key`   VARCHAR(255)  NOT NULL,
    `query_params` JSON         NOT NULL,
    `result`      MEDIUMTEXT    NOT NULL,
    `hit_count`   INT           NOT NULL DEFAULT 0,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_query_key` (`query_key`),
    KEY `idx_source` (`source`),
    KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
