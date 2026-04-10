-- Migration 022: Decision accountability log
-- Records behavioral decisions where emotion state or beliefs drove the choice.
-- Enables querying: "show decisions made under high concern", "what did high confidence change?"
-- Run via: python migrations/apply.py 022_decision_log.sql

CREATE TABLE IF NOT EXISTS samaritan_decision_log (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    turn_memory_id     INT          DEFAULT NULL     COMMENT 'ST memory row for this turn (nullable — set when known)',
    emotion_label      VARCHAR(50)  NOT NULL         COMMENT 'active emotion at decision time',
    emotion_intensity  FLOAT        NOT NULL         COMMENT '0.0-1.0 intensity',
    emotion_cluster    VARCHAR(50)  DEFAULT NULL     COMMENT 'e.g. hostile-anger, vigilant-suspicion',
    directive_weighted VARCHAR(5)   DEFAULT NULL     COMMENT 'D1, D2, D3, or D4',
    action_taken       VARCHAR(500) NOT NULL         COMMENT 'what was decided or done differently',
    belief_ids_active  JSON         DEFAULT NULL     COMMENT 'array of belief IDs that were load-bearing',
    reasoning          TEXT         DEFAULT NULL     COMMENT 'plain text causal chain: emotion/belief → decision',
    created_at         TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_created   (created_at),
    INDEX idx_emotion   (emotion_label),
    INDEX idx_cluster   (emotion_cluster),
    INDEX idx_directive (directive_weighted)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
