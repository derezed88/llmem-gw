-- 014_claude_code_target.sql
-- Add 'claude-code' as a valid target for plan steps.
-- Steps with target='claude-code' are queued for Claude Code to pick up
-- via the MCP Direct plugin, rather than being auto-executed by the
-- plan engine or waiting on a human.

ALTER TABLE {prefix}plans
  MODIFY COLUMN target ENUM('model','human','investigate','claude-code') NOT NULL DEFAULT 'model';
