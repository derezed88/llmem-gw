-- 006_plan_decomposition.sql
-- Evolve plans into a two-tier system: concept steps (human-readable intent)
-- and task steps (executable atoms with tool_call specs).
--
-- Concept steps: original natural-language plan entries (parent_id IS NULL)
-- Task steps: decomposed executable atoms (parent_id -> concept step id)
--
-- goal_id=0 allowed: plans without an explicit goal (ad-hoc plans)

-- step_type: concept = human-readable intent, task = executable atom
ALTER TABLE samaritan_plans
  ADD COLUMN step_type ENUM('concept','task') NOT NULL DEFAULT 'concept' AFTER status;

-- parent_id: NULL = concept step; non-NULL = task step pointing to its parent concept
ALTER TABLE samaritan_plans
  ADD COLUMN parent_id INT DEFAULT NULL AFTER step_type;

-- target: who/what should execute this step
--   model       = system can execute via tool_call
--   human       = requires human action
--   investigate = decomposer couldn't resolve; needs further analysis
ALTER TABLE samaritan_plans
  ADD COLUMN target ENUM('model','human','investigate') NOT NULL DEFAULT 'model' AFTER parent_id;

-- executor: specific model key assigned to execute (e.g. 'plan-executor', 'samaritan-execution')
ALTER TABLE samaritan_plans
  ADD COLUMN executor VARCHAR(64) DEFAULT NULL AFTER target;

-- tool_call: JSON spec for executable task steps
-- format: {"tool": "db_query", "args": {"query": "SELECT ..."}}
ALTER TABLE samaritan_plans
  ADD COLUMN tool_call TEXT DEFAULT NULL COMMENT 'JSON tool invocation spec' AFTER executor;

-- result: execution output or human notes
ALTER TABLE samaritan_plans
  ADD COLUMN result TEXT DEFAULT NULL COMMENT 'Execution output or human notes' AFTER tool_call;

-- approval: tracks proposal/approval lifecycle
--   proposed  = plan created, awaiting review
--   approved  = human/system approved for execution
--   rejected  = human rejected
--   auto      = auto-approved (system-generated, no gate required)
ALTER TABLE samaritan_plans
  ADD COLUMN approval ENUM('proposed','approved','rejected','auto') NOT NULL DEFAULT 'proposed' AFTER result;

-- Index for parent-child lookups
ALTER TABLE samaritan_plans
  ADD KEY idx_parent (parent_id);

-- Index for executor queue: find pending approved task steps
ALTER TABLE samaritan_plans
  ADD KEY idx_exec_queue (step_type, status, approval, target);
