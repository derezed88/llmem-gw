-- Migration 023: samaritan_sms — persistent SMS log with delivery and ack tracking
-- Replaces in-memory _inbox ring buffer in plugin_sms_proxy.py

CREATE TABLE IF NOT EXISTS samaritan_sms (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    direction       ENUM('inbound', 'outbound') NOT NULL,
    phone           VARCHAR(32)  NOT NULL,           -- sender (inbound) or recipient (outbound)
    contact_name    VARCHAR(128),                    -- resolved from person table, or NULL
    message         TEXT         NOT NULL,
    received_at     DATETIME     NOT NULL,           -- when SMS arrived / was queued for send
    delivered_at    DATETIME     DEFAULT NULL,       -- when surfaced to a chat or voice session
    delivered_to    VARCHAR(64)  DEFAULT NULL,       -- session_id or 'voice_relay'
    acked_at        DATETIME     DEFAULT NULL,       -- when admin explicitly acknowledged
    raw_payload     JSON         DEFAULT NULL,       -- original POST body for debugging
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_direction   (direction),
    INDEX idx_phone       (phone),
    INDEX idx_received_at (received_at),
    INDEX idx_acked_at    (acked_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
