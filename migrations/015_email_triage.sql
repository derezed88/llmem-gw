-- 015_email_triage.sql
-- Email triage: rules engine + decision log for autonomous email handling.
-- Phase 1: scan, classify, log decisions. No destructive actions.
-- Phase 2: rules-based auto-handling (delete, spam, archive, folder, notify).

-- Rules table: persistent classification rules (user-created, auto-proposed, claude-created)
CREATE TABLE IF NOT EXISTS {prefix}email_rules (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    rule_name   VARCHAR(100) NOT NULL,
    match_type  ENUM('sender','domain','subject','body','header') NOT NULL,
    match_value VARCHAR(255) NOT NULL,
    match_mode  ENUM('exact','contains','regex') NOT NULL DEFAULT 'contains',
    action      ENUM('delete','spam','archive','folder','notify','unsubscribe','skip') NOT NULL,
    action_args VARCHAR(255) DEFAULT NULL,
    priority    INT NOT NULL DEFAULT 50,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    hit_count   INT NOT NULL DEFAULT 0,
    last_hit_at DATETIME DEFAULT NULL,
    source      ENUM('user','auto','claude') NOT NULL DEFAULT 'user',
    notes       VARCHAR(500) DEFAULT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_enabled_priority (enabled, priority)
);

-- Triage log: every email scanned gets a decision record
CREATE TABLE IF NOT EXISTS {prefix}email_triage (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    email_uid       VARCHAR(64) NOT NULL,
    folder          VARCHAR(100) NOT NULL DEFAULT 'INBOX',
    sender          VARCHAR(255) NOT NULL,
    sender_domain   VARCHAR(100) NOT NULL,
    subject         VARCHAR(500) NOT NULL,
    received_at     DATETIME DEFAULT NULL,
    body_preview    TEXT DEFAULT NULL,
    classification  ENUM('delete','spam','archive','folder','notify','unsubscribe','skip','uncertain') NOT NULL,
    confidence      FLOAT NOT NULL DEFAULT 0.0,
    action_taken    ENUM('logged','executed','escalated','skipped') NOT NULL DEFAULT 'logged',
    matched_rule_id INT DEFAULT NULL,
    llm_reasoning   TEXT DEFAULT NULL,
    notified        BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed        BOOLEAN NOT NULL DEFAULT FALSE,
    user_override   VARCHAR(50) DEFAULT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    KEY idx_uid (email_uid),
    KEY idx_classification (classification),
    KEY idx_reviewed (reviewed),
    KEY idx_sender_domain (sender_domain)
);
