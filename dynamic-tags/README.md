# Dynamic Tags — Prompt Generator & Pusher

End-to-end tool for generating optimized text prompts for Coactive Dynamic Tags and pushing them via the v3 API.

## Features

- **Generate** — Create optimized CLIP/Qwen prompts from tag descriptions using OpenAI or Anthropic
- **Push** — Apply generated prompts to a Coactive Dynamic Tags group via the v3 API
- **Run** — End-to-end: fetch tags from API → generate prompts → push back → publish

## Requirements

```bash
pip install requests openai pydantic
# Optional: pip install anthropic  (if using --provider anthropic)
```

## Quick Start

### End-to-end (recommended)

```bash
export COACTIVE_API_TOKEN="your-coactive-token"
export OPENAI_API_KEY="your-openai-key"

python3 dynamic_tags.py run \
    --group-url "https://app.coactive.ai/dynamic-tags/groups/<gid>/versions/<vid>" \
    --modality visual \
    --publish
```

### Two-step workflow

```bash
# Step 1: Generate prompts offline
python3 dynamic_tags.py generate \
    --input my_tags.json \
    --output my_prompts.json \
    --modality visual

# Step 2: Push to Coactive
python3 dynamic_tags.py push \
    --token YOUR_TOKEN \
    --group-url "https://app.coactive.ai/dynamic-tags/groups/<gid>/versions/<vid>" \
    --prompts my_prompts.json \
    --publish
```

## Commands

| Command    | Description                                      | Requires Token | Requires LLM Key |
|------------|--------------------------------------------------|:--------------:|:-----------------:|
| `generate` | Generate prompts from a local tags JSON file     | No             | Yes               |
| `push`     | Push a prompts file to the Coactive API          | Yes            | No                |
| `run`      | End-to-end: fetch tags, generate, push, publish  | Yes            | Yes               |

## Input Formats

### Tags file (for `generate` command)

```json
{
  "group_name": "Ad Creative Tones",
  "tags": [
    {
      "name": "Inspirational and Uplifting",
      "description": "Ad creative that inspires, motivates, and evokes positive feelings"
    },
    {
      "name": "Humorous and Playful",
      "description": "Ad creative that uses humor and wit to engage viewers"
    }
  ]
}
```

### Prompts file (output from `generate`, input to `push`)

```json
{
  "results": [
    {
      "tag_name": "Inspirational and Uplifting",
      "modality": "visual",
      "prompts": [
        "inspiring ad creative",
        "motivational advertisement",
        "empowering visuals"
      ]
    }
  ]
}
```

## Options

### Common options
| Flag | Description |
|------|-------------|
| `--modality` | `visual` (CLIP-based) or `transcript` (Qwen-based). Default: `visual` |
| `--dry-run` | Preview without making changes |

### Generation options
| Flag | Description |
|------|-------------|
| `--provider` | `openai` (default) or `anthropic` |
| `--model` | Override model name (e.g., `gpt-4o-mini`) |

### Push options
| Flag | Description |
|------|-------------|
| `--token` | Coactive personal token (or `COACTIVE_API_TOKEN` env var) |
| `--group-url` | Coactive UI URL (extracts group/version IDs automatically) |
| `--group-id` / `--version-id` | Explicit UUIDs (alternative to URL) |
| `--publish` | Auto-publish the version after pushing prompts |

## Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `COACTIVE_API_TOKEN` | `push`, `run` | Coactive personal API token |
| `OPENAI_API_KEY` | `generate`, `run` | OpenAI API key |
| `ANTHROPIC_API_KEY` | `generate`, `run` (with `--provider anthropic`) | Anthropic API key |

## API Details

Uses the Coactive Dynamic Tags **v3 API**:
- Auth: `POST /api/v0/login` (personal token → JWT exchange)
- List tags: `GET /api/v3/dynamic-tags/groups/{gid}/versions/{vid}/tags`
- Update tag: `PATCH /api/v3/dynamic-tags/groups/{gid}/tags/{tid}/versions/{tvid}`
- Publish: `POST /api/v3/dynamic-tags/groups/{gid}/versions/{vid}/publish`

## Author

Madan — Coactive AI
