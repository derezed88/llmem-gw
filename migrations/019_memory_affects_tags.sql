-- Migration: 019_memory_affects_tags
-- Adds schedule-impact annotation columns to short and long-term memory tables.
-- affects_tags: JSON array of routine tags this memory impacts (e.g. ["lee-gym","lee-pickup"])
-- effect_duration_days: how many days the effect lasts (NULL = unknown/permanent)
-- effect_recurs: recurrence cadence ("monthly","weekly","one-time", NULL = unknown)
-- Run via: python migrations/apply.py 019_memory_affects_tags.sql

ALTER TABLE `{prefix}memory_shortterm`
    ADD COLUMN `affects_tags`        JSON         DEFAULT NULL COMMENT 'routine tags this memory impacts',
    ADD COLUMN `effect_duration_days` SMALLINT    DEFAULT NULL COMMENT 'how long the effect lasts in days',
    ADD COLUMN `effect_recurs`       VARCHAR(50)  DEFAULT NULL COMMENT 'recurrence: monthly|weekly|one-time|etc';

ALTER TABLE `{prefix}memory_longterm`
    ADD COLUMN `affects_tags`        JSON         DEFAULT NULL COMMENT 'routine tags this memory impacts',
    ADD COLUMN `effect_duration_days` SMALLINT    DEFAULT NULL COMMENT 'how long the effect lasts in days',
    ADD COLUMN `effect_recurs`       VARCHAR(50)  DEFAULT NULL COMMENT 'recurrence: monthly|weekly|one-time|etc';
