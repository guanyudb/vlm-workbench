-- Lakebase schema for the Surgical VLM Workbench (Phase 2)
--
-- Apply with:  databricks lakebase psql --instance <name> --profile vlm < schema.sql
--
-- This schema is NOT deployed yet. It documents the persistence layer that
-- Compare / Eval / Jobs surfaces will use in Phase 2.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── Prompts ───────────────────────────────────────────────────────────
-- Versioned prompt library. parent_id chains revisions; tags enable filtering.
CREATE TABLE IF NOT EXISTS prompts (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id     UUID REFERENCES prompts(id) ON DELETE SET NULL,
    name          TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    body          TEXT NOT NULL,
    vocabulary    TEXT[],
    tags          TEXT[],
    notes         TEXT,
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);
CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);
CREATE INDEX IF NOT EXISTS idx_prompts_tags ON prompts USING gin(tags);

-- ── Snapshots ────────────────────────────────────────────────────────
-- Persisted Playground runs: (frames × models × prompt) → results.
CREATE TABLE IF NOT EXISTS snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    video_path      TEXT,
    frame_paths     TEXT[] NOT NULL,
    model_endpoints TEXT[] NOT NULL,
    prompt_id       UUID REFERENCES prompts(id) ON DELETE SET NULL,
    prompt_body     TEXT NOT NULL,        -- denormalised — keeps history readable if prompt is deleted
    results         JSONB NOT NULL,
    cost_usd        NUMERIC(10, 4),
    latency_ms      INTEGER,
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_by_at ON snapshots(created_by, created_at DESC);

-- ── Golden labels ────────────────────────────────────────────────────
-- Frame-level ground truth for the Eval surface.
CREATE TABLE IF NOT EXISTS golden_labels (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    frame_path        TEXT NOT NULL,
    primary_class     TEXT NOT NULL,
    secondary_classes TEXT[],
    anatomy_notes     TEXT,
    confidence        NUMERIC(3, 2),
    labeled_by        TEXT NOT NULL,
    labeled_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (frame_path, labeled_by)
);
CREATE INDEX IF NOT EXISTS idx_golden_labels_frame ON golden_labels(frame_path);

-- ── Batch jobs ───────────────────────────────────────────────────────
-- Configurations for the Jobs surface — what to run, when, where to write.
CREATE TABLE IF NOT EXISTS batch_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    model_endpoints TEXT[] NOT NULL,
    prompt_id       UUID REFERENCES prompts(id) ON DELETE RESTRICT,
    input_filter    TEXT NOT NULL,        -- SQL fragment selecting frames
    output_table    TEXT NOT NULL,
    cron            TEXT,
    last_run_id     TEXT,
    status          TEXT NOT NULL DEFAULT 'idle',
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Users (lightweight, derived from OAuth in a later phase) ─────────
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    role        TEXT NOT NULL DEFAULT 'researcher', -- researcher | surgeon | operator | admin
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
