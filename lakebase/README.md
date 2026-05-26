# Lakebase setup (Phase 2)

Lakebase is the persistence layer for the workbench's Compare / Eval / Jobs
surfaces (saved prompts, snapshots, golden labels, batch-job configs).

## What's here

- `schema.sql` — Postgres schema with five tables: `prompts`, `snapshots`,
  `golden_labels`, `batch_jobs`, `users`.

## When to provision

Phase 1 (Playground only) doesn't need Lakebase — every run is ephemeral.

When you're ready for Phase 2 (Compare / Eval / Jobs):

```bash
# 1. Create a Lakebase instance in the vlm workspace
databricks --profile=vlm lakebase create-database-instance \
  --name vlm-workbench \
  --capacity CU_1

# 2. Note the instance ID; uncomment the `lakebase` resource in app.yaml
#    and fill in the instance ID + permission

# 3. Apply the schema
databricks --profile=vlm lakebase psql --instance vlm-workbench < schema.sql
```

The app's service principal automatically gets credentials for instances
declared in `app.yaml` — no manual secret wiring required.

## Notes

- The schema uses `pgcrypto` for `gen_random_uuid()`. `CREATE EXTENSION IF
  NOT EXISTS pgcrypto;` is at the top.
- `snapshots.results` is `JSONB` for flexibility — future surfaces can index
  inside it without migrations.
- `prompts` are versioned via `(name, version)` unique. To "edit" a prompt,
  insert a new row with the same name + incremented version + parent_id of
  the previous row.
