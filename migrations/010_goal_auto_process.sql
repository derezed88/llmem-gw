-- 010_goal_auto_process.sql
-- Add autonomous goal processing status to goals table.
--
-- auto_process_status tracks the lifecycle of autonomously-managed goals:
--   NULL       = not yet seen by goal_processor (eligible for scanning)
--   proposed   = plan proposed to user, awaiting response
--   approved   = user approved, ready for execution
--   deferred   = user said "not now" — re-check after defer_until
--   rejected   = user said "no" — never auto-process
--   executing  = serial execution in progress
--   paused_user = blocked on a user-owned step
--   completed  = all steps done
--
-- defer_until: when a deferred goal becomes eligible for re-proposal

ALTER TABLE {prefix}goals
  ADD COLUMN auto_process_status ENUM(
    'proposed','approved','deferred','rejected',
    'executing','paused_user','completed'
  ) DEFAULT NULL AFTER abandon_reason;

ALTER TABLE {prefix}goals
  ADD COLUMN defer_until DATETIME DEFAULT NULL AFTER auto_process_status;

-- Index for goal_processor scanning
ALTER TABLE {prefix}goals
  ADD KEY idx_auto_process (status, auto_process_status);
