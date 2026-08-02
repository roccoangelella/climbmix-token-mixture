"""Pinned source identity and calibration constants."""

DATASET_REPOSITORY = "nvidia/Nemotron-ClimbMix"
DATASET_REVISION = "5eaa64b9c0c85b7f56af01d7dffdb0795816b12b"
SOURCE_DATA_GLOB = "part_*.tokenized.jsonl"
SELECTION_SEED = "small-llm-climbmix-production-v1"

ACCEPTED_CLUSTER_IDS = frozenset(range(1, 11)) | frozenset(range(12, 21))
EXCLUDED_CLUSTER_IDS = frozenset({11})
ALL_CLUSTER_IDS = frozenset(range(1, 21))

TOKEN_MIN = 0
TOKEN_MAX = 50256
REGION_BYTES = 256 * 1024 * 1024
BOUNDARY_SCAN_CHUNK_BYTES = 4 * 1024 * 1024
FORWARD_FETCH_CHUNK_BYTES = 8 * 1024 * 1024

HF_HUB_BASE = "https://huggingface.co"
RESOLVE_URL_TEMPLATE = HF_HUB_BASE + "/datasets/{repository}/resolve/{revision}/{path}"
TREE_URL_TEMPLATE = HF_HUB_BASE + "/api/datasets/{repository}/tree/{revision}"
HTTP_TIMEOUT_SECONDS = 60.0
HTTP_MAX_RETRIES = 6
HTTP_BACKOFF_BASE_SECONDS = 1.5
HTTP_BACKOFF_MAX_SECONDS = 30.0
HTTP_USER_AGENT = "climbmix-token-mixture/1.0"

WORK_PLAN_FILENAME = "work_plan.json"
WORK_PLAN_SCHEMA_VERSION = 2
