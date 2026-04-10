-- Migration: 020_turn_group_id
-- Adds turn_group_id to short-term memory so multi-topic split rows stay linkable.
-- When a conversation turn is split into multiple topic segments, all resulting
-- memory rows share the same turn_group_id (UUID) for reconstruction.
-- Run via: python migrations/apply.py 020_turn_group_id.sql

ALTER TABLE `{prefix}memory_shortterm`
    ADD COLUMN `turn_group_id` VARCHAR(36) DEFAULT NULL COMMENT 'UUID linking rows from the same conversation turn (multi-topic splits)',
    ADD INDEX  `idx_turn_group_id` (`turn_group_id`);
