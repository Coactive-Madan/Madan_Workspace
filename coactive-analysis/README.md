# Coactive Dynamic Tags & Concepts Analyzer

A command-line tool for analyzing the performance of Coactive Dynamic Tags and Concepts. Generates comprehensive statistics, score distributions, and markdown reports.

## Features

- **Analyze Concepts**: Run analysis on all concepts in a dataset
- **Analyze Dynamic Tags**: Run analysis on specific dynamic tag groups
- **Auto-detect from URL**: Provide a Coactive URL and the tool will detect the resource type
- **Markdown Reports**: Generate formatted markdown reports with statistics
- **JSON Output**: Export raw data for further processing
- **Score Distribution**: View score buckets (>0.9, 0.8-0.9, 0.7-0.8, etc.)
- **Top Images**: Identify highest-scoring images per concept/tag

## Installation

### Prerequisites

- Python 3.8+
- `requests` library

### Setup

```bash
# Clone the repository
git clone https://github.com/Coactive-Madan/Madan_Workspace.git
cd Madan_Workspace/coactive-analysis

# Install dependencies
pip install requests
```

## Usage

### Basic Commands

```bash
# Analyze all concepts
python coactive_analyzer.py --token YOUR_API_TOKEN --type concepts

# Analyze a specific dynamic tag group
python coactive_analyzer.py --token YOUR_API_TOKEN --type dynamic-tags --group-id GROUP_ID

# Analyze from URL (auto-detects type)
python coactive_analyzer.py --token YOUR_API_TOKEN --url "https://app.coactive.ai/concepts?page=1"

# Analyze dynamic tags from URL
python coactive_analyzer.py --token YOUR_API_TOKEN --url "https://app.coactive.ai/dynamic-tag-groups/0cbad66d-388a-461b-9cd3-8d1ac569e36f"
```

### Output Options

```bash
# Save report to file
python coactive_analyzer.py --token YOUR_TOKEN --type concepts --output report.md

# Output as JSON
python coactive_analyzer.py --token YOUR_TOKEN --type concepts --json

# Save JSON to file
python coactive_analyzer.py --token YOUR_TOKEN --type concepts --json --output results.json
```

### Command-Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--token` | `-t` | Yes | Coactive API token |
| `--type` | | No* | Resource type: `concepts` or `dynamic-tags` |
| `--url` | `-u` | No* | Coactive URL (auto-detects type) |
| `--group-id` | `-g` | No** | Dynamic tag group ID |
| `--dataset-id` | `-d` | No | Dataset ID (auto-detected if not provided) |
| `--output` | `-o` | No | Output file path |
| `--json` | | No | Output raw JSON instead of markdown |

\* Either `--type` or `--url` must be provided
\** Required when `--type` is `dynamic-tags`

## Getting Your API Token

1. Log in to Coactive at https://app.coactive.ai
2. Navigate to Settings > API Tokens
3. Create a new token or copy an existing one

## Example Output

### Statistics Table

| Concept | Count | Avg Score | Min | Max | Median | Std Dev |
|---------|------:|----------:|----:|----:|-------:|--------:|
| Interest Video Gaming | 1,028 | 0.0755 | 0.0 | 0.9903 | 0.0041 | 0.2089 |
| Interest Television | 1,028 | 0.0688 | 0.0 | 0.9921 | 0.0055 | 0.1884 |

### Score Distribution

| Concept | >0.9 | 0.8-0.9 | 0.7-0.8 | 0.5-0.7 | <0.5 |
|---------|-----:|--------:|--------:|--------:|-----:|
| Interest Video Gaming | 24 | 20 | 9 | 16 | 959 |
| Interest Television | 24 | 12 | 2 | 13 | 977 |

### High Confidence Summary

| Concept | Count (>=0.8) | % of Total |
|---------|-------------:|-----------:|
| Interest Video Gaming | 44 | 4.3% |
| Interest Television | 36 | 3.5% |

## How It Works

### Analysis Methodology

1. **Authentication**: Exchanges your API token for a JWT access token
2. **Discovery**: Finds available SQL tables for concepts or dynamic tags
3. **Statistics Query**: Runs aggregation queries to compute:
   - Total images per concept/tag
   - Average, min, max, median scores
   - Standard deviation
   - Score distribution across thresholds
4. **Top Images**: Identifies the highest-scoring images for each concept/tag
5. **Report Generation**: Formats results as markdown or JSON

### SQL Tables

- **Concepts**: Each concept has its own table: `concept_{concept_name}`
- **Dynamic Tags**: All tags in a group share one table: `group_{group_name}_visual`

### Key Metrics

- **Avg Score**: Mean confidence score across all images
- **High Confidence (>=0.8)**: Images that confidently match the concept/tag
- **Score Distribution**: Breakdown by threshold buckets for precision/recall tuning

## Troubleshooting

### Authentication Failed

```
❌ Authentication failed: 401
```

- Verify your API token is correct
- Ensure the token has not expired
- Check that the token has access to the target dataset

### No Tables Found

```
❌ No concept tables found
```

- Concepts/tags may not have been indexed yet
- Verify the dataset ID is correct
- Check that you have access to the dataset

### Query Timeout

```
❌ Query timed out
```

- Large datasets may take longer to process
- Try analyzing a smaller subset
- Check your network connection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Internal use only - Coactive AI

## Support

For issues or questions, contact the Coactive team or open an issue on GitHub.
