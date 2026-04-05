-- Migration 018: Remove xAI STT prosody fields from samaritan_emotions
-- These columns were added for xAI emotional voice mode (xai/xai-b mic modes),
-- which has been removed. Emotion assessment is now handled solely by Claude
-- using the full conversational context.
--
-- Note: Uses DROP COLUMN IF EXISTS (MySQL 8.0.29+). Safe to run even if
-- columns were never added to this instance.

ALTER TABLE samaritan_emotions
    DROP COLUMN IF EXISTS xai_emotion_label,
    DROP COLUMN IF EXISTS xai_confidence,
    DROP COLUMN IF EXISTS xai_prosody;
