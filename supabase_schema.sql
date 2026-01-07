-- ==============================================================================
-- AUTONOMOUS RESEARCH ENGINE MEMORY SCHEMA (REFINED)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Run Memory (Short-Term / Audit Log)
-- Purpose: Faithfully record every iteration outcome for debugging and audit.
-- Properties: Immutable, Append-Only.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,                        -- Unique ID for the specific execution run
    iteration INT NOT NULL,                      -- Iteration index (0-based)
    research_goal TEXT NOT NULL,                 -- Original user prompt
    execution_mode TEXT NOT NULL,                -- 'validation' | 'scientific'
    experiment_spec JSONB NOT NULL,              -- Full JSON specification
    evaluation_verdict JSONB NOT NULL,           -- Full JSON verdict from Evaluation Agent
    classification TEXT NOT NULL,                -- 'robust' | 'promising' | 'spurious' | 'failed'
    issue_type TEXT NOT NULL,                    -- 'design' | 'data' | 'execution' | 'none'
    feedback_passed TEXT,                        -- Feedback sent to next iteration (Lineage)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints for Data Integrity
    CONSTRAINT run_memory_execution_mode_check CHECK (execution_mode IN ('validation', 'scientific')),
    CONSTRAINT run_memory_classification_check CHECK (classification IN ('robust', 'promising', 'spurious', 'failed')),
    CONSTRAINT run_memory_issue_type_check CHECK (issue_type IN ('design', 'data', 'execution', 'none'))
);

-- Index for fast lookup by run_id
CREATE INDEX IF NOT EXISTS idx_run_memory_run_id ON run_memory(run_id);

-- ------------------------------------------------------------------------------
-- 2. Knowledge Memory (Long-Term / Canonical)
-- Purpose: Store ONLY validated, high-confidence scientific conclusions.
-- Properties: High signal-to-noise ratio. No failed experiments.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,                        -- Link to the run that produced this knowledge
    research_question TEXT NOT NULL,             -- Final specific question answered
    final_hypothesis TEXT NOT NULL,              -- The hypothesis that was supported
    decision TEXT NOT NULL,                      -- MUST be 'Reject H0'
    effect_summary TEXT NOT NULL,                -- Human-readable finding
    statistical_summary JSONB NOT NULL,          -- { test: ..., p_value: ..., effect_size: ... }
    assumptions_status JSONB NOT NULL,           -- { assumption_check: 'PASS', ... }
    execution_mode TEXT NOT NULL,                -- 'validation' | 'scientific'
    confidence_level TEXT NOT NULL,              -- 'high' | 'medium'
    supporting_context JSONB,                    -- { model_family, data_modality, task_type, etc. }
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Strict Scientific Guardrails
    CONSTRAINT knowledge_decision_check CHECK (decision = 'Reject H0'),
    CONSTRAINT knowledge_confidence_check CHECK (confidence_level IN ('high', 'medium'))
);

-- Index for searching by topic/question
CREATE INDEX IF NOT EXISTS idx_knowledge_memory_question ON knowledge_memory USING gin(to_tsvector('english', research_question));

-- Ensure one knowledge entry per run (Scientific Logic)
CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_per_run ON knowledge_memory(run_id);

-- Note: FK constraint from knowledge_memory(run_id) to run_memory(run_id) is NOT added
-- because run_memory.run_id is not unique (1 run = many iterations).
