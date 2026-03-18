-- Migration: 011_ged_progress
-- Creates GED tutoring progress tables for all GED subject databases.
-- Tables: ged_topic_scores, ged_quiz_results, ged_session_state
-- Apply with: python migrations/apply.py 011_ged_progress.sql
-- Safe to apply to all databases — non-GED DBs get the tables with their prefix,
-- but they are never queried (only GED model DBs use them).

-- ============================================================
-- GED TOPIC SCORES
-- Tracks Lee's strength per GED topic area, 0.0-1.0.
-- One row per topic (UNIQUE KEY). Score updated after each quiz.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}ged_topic_scores` (
    `id`         INT(11)  NOT NULL AUTO_INCREMENT,
    `topic`      VARCHAR(64) NOT NULL,
    `score`      FLOAT    NOT NULL DEFAULT 0.5,
    `attempts`   INT      NOT NULL DEFAULT 0,
    `correct`    INT      NOT NULL DEFAULT 0,
    `phase`      ENUM('assessment','lesson','quiz','exam') DEFAULT 'assessment',
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_topic` (`topic`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- GED QUIZ RESULTS
-- Per-question record for every assessment, quiz, and exam question.
-- Used for analytics and session recovery.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}ged_quiz_results` (
    `id`          INT(11)  NOT NULL AUTO_INCREMENT,
    `session_id`  VARCHAR(255) DEFAULT NULL,
    `phase`       ENUM('assessment','quiz','exam') NOT NULL,
    `topic`       VARCHAR(64)  NOT NULL,
    `question`    TEXT         NOT NULL,
    `lee_answer`  TEXT         DEFAULT NULL,
    `correct`     TINYINT(1)   DEFAULT NULL,
    `explanation` TEXT         DEFAULT NULL,
    `created_at`  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic`   (`topic`),
    KEY `idx_phase`   (`phase`),
    KEY `idx_session` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- GED SESSION STATE
-- Key-value store for course phase, loop count, readiness flag.
-- Persists state across sessions so Lee can pick up where she left off.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}ged_session_state` (
    `id`         INT(11)     NOT NULL AUTO_INCREMENT,
    `key`        VARCHAR(64) NOT NULL,
    `value`      TEXT        NOT NULL,
    `updated_at` TIMESTAMP   DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_key` (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
