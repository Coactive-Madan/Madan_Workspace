# Celebrity Detection + Dynamic Tags / Concepts Workflow

A reusable workflow for leveraging Coactive's celebrity detection capabilities. Enroll celebrities, detect their faces across your dataset, and join with Dynamic Tags (DT) or Concept scores -- **entirely via the Coactive API** (no internal infrastructure access required).

## Architecture Overview

```
                        ┌─────────────────────┐
                        │  Seed Images (5-18)  │
                        └─────────┬───────────┘
                                  │
                        ┌─────────▼───────────┐
                        │  Enrollment Script   │
                        │  (4-step API flow)   │
                        └─────────┬───────────┘
                                  │ ~1-2 hr backfill
                        ┌─────────▼───────────┐
                        │  Face Detections     │
                        │  (Celebrity Det API) │
                        └─────────┬───────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                                       │
    ┌─────────▼──────────┐              ┌─────────────▼─────────┐
    │  Concepts           │              │  Dynamic Tags          │
    │  coactive_table_adv │              │  SQL: dt_[name]_visual │
    │  (Query Engine SQL) │              │  (Query Engine SQL)    │
    └─────────┬──────────┘              └─────────────┬─────────┘
              │                                       │
              │                              SQL fails? (table not found)
              │                                       │
              │                              ┌────────▼────────┐
              │                              │  Auto-Discover   │
              │                              │  DT Groups (API) │
              │                              └────────┬────────┘
              │                                       │
              │                              ┌────────▼────────┐
              │                              │  DT API Fallback │
              │                              │  scoring-preview │
              │                              │  + asset-check   │
              │                              └────────┬────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                        ┌─────────▼───────────┐
                        │  Join on image_id    │
                        │  Analytics + Queries │
                        └─────────────────────┘
```

## Prerequisites

1. **Coactive API key** -- from your Coactive account
2. **Dataset** -- your image/video dataset ingested in Coactive
3. **Enrolled celebrities** -- with backfill complete (~1-2 hours)
4. **Concepts and/or Dynamic Tags** -- created and published on your dataset in the Coactive UI

No Databricks, S3 access, or internal infrastructure required. The notebook runs anywhere with Python + `requests`.

## Quick Start

### 1. Enroll Celebrities

Clone the enrollment tools:
```bash
git clone https://github.com/Coactive-Madan/Madan_Workspace.git
cd Madan_Workspace/celebrity-enrollment
```

Create a `config.json`:
```json
{
  "client_name": "your-client",
  "coactive_api_key": "your-api-key",
  "base_url": "https://app.coactive.ai",
  "dataset_id": "your-dataset-uuid",
  "seed_images_dir": "./seed_images",
  "celebrities": [
    {
      "name": "Celebrity Name",
      "aliases": ["Alias 1", "Alias 2"],
      "image_pattern": "celebrity_name_*.jpg",
      "max_images": 18
    }
  ]
}
```

Prepare seed images (5-18 per person, JPEG/PNG, clear well-lit face shots with varied angles/lighting/expressions), then run:

```bash
python celebrity_enrollment.py
```

The enrollment script performs 4 API steps:
1. **Create draft person** -- `POST /api/v0/celebrity-detection/enroll`
2. **Upload seed images** -- `POST /api/v0/celebrity-detection/upload`
3. **Attach images + aliases** -- `PATCH /api/v0/celebrity-detection/enroll/{person_id}`
4. **Finalize** -- `POST /api/v0/celebrity-detection/enroll/{person_id}/finalize`

After finalization, backfill runs for ~1-2 hours. Verify enrollment:
```bash
python celebrity_enrollment.py --list
```

### 2. Create Dynamic Tags / Concepts

In the Coactive UI:

**For Dynamic Tags:**
1. Navigate to your dataset
2. Create a DT Group (e.g., "Broadcast Context Classification")
3. Add tags with text descriptions (e.g., "Highlight Moment", "Close-Up Shot")
4. Publish the group to trigger scoring
5. Find the **SQL table name** for your group: `dt_[group_name]_visual` (lowercase, spaces → underscores). You can discover tables by running `SHOW TABLES` in the Coactive SQL editor.

**For Concepts:**
1. Navigate to your dataset in the Coactive UI
2. Create concepts and train them with labeled examples
3. Find the concept's **probability column name** in `coactive_table_adv` -- it follows the pattern `<concept_name>_prob` (e.g., `baton_prob`). You can discover columns by running `SELECT * FROM coactive_table_adv LIMIT 1` in the Coactive SQL editor.

### 3. Run the Notebook

Import `Celebrity_Detection_DT_Workflow.ipynb` into your environment (Databricks, Jupyter, Colab, etc.) and fill in the configuration cell:

| Parameter | Description | Required? |
|-----------|-------------|-----------|
| `COACTIVE_API_KEY` | Your Coactive API key | Yes |
| `DATASET_ID` | Your dataset UUID | Yes |
| `CELEBRITIES` | Dict of `{"Name": "person_id"}` from enrollment | Yes |
| `CONCEPTS` | Dict of `{"Display Name": "sql_column_name"}` (e.g., `{"Baton": "baton_prob"}`) | If using Concepts |
| `DT_SQL_TABLES` | Dict of `{"Display Name": "sql_table_name"}` (e.g., `{"Sports": "dt_sports_visual"}`) | If using DTs (recommended) |
| `DT_GROUPS` | List of `{name, group_id, group_version_id}` | If using DTs (API-only, optional) |

You need at least one of `CONCEPTS`, `DT_SQL_TABLES`, or `DT_GROUPS` configured.

**Note:** If `DT_SQL_TABLES` is configured but a table returns `TABLE_OR_VIEW_NOT_FOUND`, the notebook automatically discovers the corresponding DT group via the API and falls back to API-based scoring. You don't need to manually configure `DT_GROUPS` as a backup.

## SQL Table Schemas

### Concepts: `coactive_table_adv` (pivoted)

Each trained concept gets its own probability column:

| Column | Type | Description |
|--------|------|-------------|
| `coactive_image_id` | string | Unique image/keyframe ID |
| `<concept>_prob` | float | Probability score 0-1 (e.g., `baton_prob`, `logo_prob`) |

```sql
SELECT coactive_image_id, baton_prob, logo_prob
FROM coactive_table_adv
WHERE coactive_image_id IN ('id1', 'id2', ...)
  AND (baton_prob > 0.5 OR logo_prob > 0.5)
```

### Dynamic Tags: `dt_[group_name]_visual` (normalized)

Each DT group gets its own table with a normalized schema:

| Column | Type | Description |
|--------|------|-------------|
| `coactive_image_id` | string | Unique image/keyframe ID |
| `tag_name` | string | The tag label (e.g., "Highlight Moment") |
| `score` | float | DT score 0-1 |

```sql
SELECT coactive_image_id, tag_name, score
FROM dt_sports_classification_visual
WHERE coactive_image_id IN ('id1', 'id2', ...)
  AND score > 0.5
```

The table name is derived from the DT group name: lowercased, spaces replaced by underscores, prefixed with `dt_` and suffixed with `_visual`.

## How Each Score Source Works

### Concepts via Query Engine (SQL API)

The notebook uses the Coactive Query Engine to fetch scores:
1. **Submit SQL** -- `POST /api/v1/queries` with a SQL query
2. **Poll status** -- `GET /api/v1/queries/{query_id}` until complete
3. **Download results** -- `GET /api/v1/queries/{query_id}/results` → returns presigned `download_url` for CSV

The notebook then "unpivots" the columnar results so each (image, concept) pair becomes a row in the joined dataset. Queries are batched (200 keyframes/batch) to avoid SQL length limits.

### Dynamic Tags via SQL (Recommended)

Uses the same Query Engine workflow as Concepts, but queries `dt_[group_name]_visual` tables. Since DT tables have a normalized schema (one row per image-tag pair), no unpivoting is needed -- the results map directly to the joined dataset.

**Important:** DT SQL tables may appear in `SHOW TABLES` but return `TABLE_OR_VIEW_NOT_FOUND` errors if the DT group is unpublished, still scoring, or not yet fully materialized. The notebook handles this automatically (see below).

### Dynamic Tags via API (Automatic Fallback)

The notebook automatically falls back to the DT API if SQL tables are unavailable. The fallback flow is:

1. **SQL query fails** with `TABLE_OR_VIEW_NOT_FOUND`
2. **Auto-discovery** -- the notebook calls `GET /api/v3/dynamic-tags/groups` to discover all DT groups, extracting `group_id` and `group_version_id` for each
3. **Name matching** -- maps the failed SQL table names back to the discovered API groups
4. **Two-phase API scoring:**
   - **Phase 1: scoring-preview** -- `GET .../scoring-preview/image-and-keyframe` returns actual DT scores (0-1 normalized), but limited to ~100 top keyframes per tag
   - **Phase 2: asset-check** -- `GET .../asset-check` returns raw cosine similarities (~0.1-0.3 range) for all remaining (keyframe, tag) pairs not covered by Phase 1, parallelized with 10 workers

This means you can configure `DT_SQL_TABLES` and the notebook will "just work" -- if the SQL tables exist, it uses them directly; if not, it seamlessly falls back to the API.

You can also set `DT_GROUPS` directly (with `group_id` and `group_version_id`) to skip SQL entirely, or leave both empty and the notebook will auto-discover all available DT groups from the API.

### Join Key

All approaches join on `image_id`:
- Face detections return `image_id` (the keyframe where the celebrity was detected)
- Concept/DT scores are keyed by `coactive_image_id` / `asset_id`
- Inner join produces rows where a celebrity face AND a score exist on the same keyframe

## Output

The notebook produces:
- **Per-celebrity stats** -- face counts, avg confidence, avg scores
- **Classification cross-tab** -- Celebrity x Tag/Concept matrix
- **Content mix breakdown** -- percentage distribution per celebrity
- **Coactive UI SQL queries** -- paste into the Coactive platform to visually browse matched keyframes
- **Content profile queries** -- best keyframe per tag for each celebrity
- **CSV/DataFrame export** -- for downstream analysis

## API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v0/login` | POST | Exchange API key for JWT |
| `/api/v1/queries` | POST | Submit SQL query (async) |
| `/api/v1/queries/{query_id}` | GET | Check query status |
| `/api/v1/queries/{query_id}/results` | GET | Get presigned download URL for CSV results |
| `/api/v0/celebrity-detection/enroll` | POST | Create draft person |
| `/api/v0/celebrity-detection/upload` | POST | Upload seed images |
| `/api/v0/celebrity-detection/enroll/{id}` | PATCH | Attach images/aliases |
| `/api/v0/celebrity-detection/enroll/{id}/finalize` | POST | Trigger backfill |
| `/api/v0/celebrity-detection/persons` | GET | List enrolled persons |
| `/api/v0/celebrity-detection/faces-with-person` | GET | Get face detections |
| `/api/v3/dynamic-tags/groups` | GET | Auto-discover DT groups (used by fallback) |
| `/api/v3/dynamic-tags/groups/{id}/versions/{vid}/tags` | GET | List DT tags (API fallback) |
| `.../scoring-preview/image-and-keyframe` | GET | DT scores 0-1 (API fallback) |
| `.../asset-check` | GET | Raw cosine similarities (API fallback) |

## Thresholds

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Face confidence | > 0.7 | High-quality face matches |
| Concept probability (`*_prob` columns) | > 0.5 | 0-1 probability from `coactive_table_adv` |
| DT score (SQL or scoring-preview) | > 0.5 | Normalized 0-1 scores |
| DT score (asset-check API fallback) | > 0.19 | Raw cosine similarities have a lower range |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 0 faces returned | Backfill may not be complete. Wait 1-2 hours after finalization. |
| DT SQL table not found | The notebook auto-falls back to the DT API. If fallback also fails, verify the DT group exists in the Coactive UI. Note: tables may exist in `SHOW TABLES` but still return `TABLE_OR_VIEW_NOT_FOUND` if the group is unpublished or not yet materialized. |
| No DT scores returned via SQL | Check that the DT group is published and scored. Table name format: `dt_[group_name]_visual`. The notebook will automatically try the API fallback if SQL fails. |
| DT auto-discovery finds no groups | Verify your API key has access to DT groups. Check `GET /api/v3/dynamic-tags/groups` manually. |
| Query Engine timeout | Increase `max_wait` in `poll_query()`. Large datasets take longer. |
| JWT expired | Re-run the authentication cell. JWTs are short-lived. |
| `person_id` not found | Run `python celebrity_enrollment.py --list` to verify enrollment. |
| No concept scores returned | Verify concept is trained and published. Check column name matches in `coactive_table_adv`. |
| Low scoring-preview overlap | Only applies to API fallback. scoring-preview returns ~100 keyframes/tag; asset-check covers the rest. |

## File Structure

```
celebrity-detection-workflow/
  Celebrity_Detection_DT_Workflow.ipynb  # Main reusable notebook
  README.md                              # This file
  config.example.json                    # Config template for enrollment
```

## Enrollment Script Reference

The enrollment tools live at: [github.com/Coactive-Madan/Madan_Workspace/tree/main/celebrity-enrollment](https://github.com/Coactive-Madan/Madan_Workspace/tree/main/celebrity-enrollment)

Key files:
- `celebrity_enrollment.py` -- Generic enrollment script (4-step workflow)
- `celeb_dt_workflow.py` -- Post-enrollment face + DT workflow (CLI version)
- `config.example.json` -- Configuration template
