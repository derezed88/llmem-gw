-- 005_goal_health.sql
-- Add attempt tracking columns for goal health / quit logic.
-- attempt_count: incremented by reflection when goal is discussed/worked on
-- failure_count: incremented when same-slug self-failure rows accumulate
-- abandon_reason: set when goal is auto-abandoned by goal_health pass

ALTER TABLE samaritan_goals
  ADD COLUMN attempt_count  INT NOT NULL DEFAULT 0 AFTER memory_link,
  ADD COLUMN failure_count  INT NOT NULL DEFAULT 0 AFTER attempt_count,
  ADD COLUMN abandon_reason TEXT DEFAULT NULL AFTER failure_count;

-- Also add to qwen and test DB tables if they exist
-- (safe to fail if tables don't exist in those DBs)
