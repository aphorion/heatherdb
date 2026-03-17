"""Task definitions for specialist training and hybrid evaluation."""


CODING_TASKS = [
    {
        "id": "code_01",
        "description": "Implement a function `merge_sorted(a, b)` that merges two sorted lists into one sorted list in O(n+m) time.",
        "domain": "coding",
    },
    {
        "id": "code_02",
        "description": "Fix the bug in this binary search: it returns -1 for the last element. The issue is using `high = len(arr) - 1` with `while low < high` instead of `while low <= high`.",
        "domain": "coding",
    },
    {
        "id": "code_03",
        "description": "Refactor this function that has 5 nested if-else blocks for validating user input into a clean, early-return guard clause pattern.",
        "domain": "coding",
    },
    {
        "id": "code_04",
        "description": "Write unit tests for a `Stack` class that supports push, pop, peek, and is_empty. Cover edge cases: empty pop, single element, large stack.",
        "domain": "coding",
    },
    {
        "id": "code_05",
        "description": "Implement a least-recently-used (LRU) cache with O(1) get and put using a dict and doubly-linked list.",
        "domain": "coding",
    },
    {
        "id": "code_06",
        "description": "Optimize this O(n^2) function that finds duplicate elements in a list. Use a set for O(n) performance.",
        "domain": "coding",
    },
    {
        "id": "code_07",
        "description": "Implement a `flatten(nested_list)` function that recursively flattens arbitrarily nested lists into a single flat list.",
        "domain": "coding",
    },
    {
        "id": "code_08",
        "description": "Write a decorator `@retry(max_attempts=3, delay=1.0)` that retries a function on exception with exponential backoff.",
        "domain": "coding",
    },
    {
        "id": "code_09",
        "description": "Implement a simple tokenizer that splits text into words, handles punctuation, and normalizes whitespace. Return a list of tokens.",
        "domain": "coding",
    },
    {
        "id": "code_10",
        "description": "Fix the race condition in this producer-consumer pattern where the consumer sometimes reads stale data. Use proper locking.",
        "domain": "coding",
    },
    {
        "id": "code_11",
        "description": "Implement `group_by(items, key_fn)` that groups a list of items by the result of key_fn, returning a dict of lists.",
        "domain": "coding",
    },
    {
        "id": "code_12",
        "description": "Write a function that validates an email address using regex, checking for proper format with @ symbol, domain, and TLD.",
        "domain": "coding",
    },
    {
        "id": "code_13",
        "description": "Implement a priority queue using a binary heap with insert, extract_min, and peek operations.",
        "domain": "coding",
    },
    {
        "id": "code_14",
        "description": "Refactor this class with 12 methods into two smaller classes using the Single Responsibility Principle. One handles data, one handles formatting.",
        "domain": "coding",
    },
    {
        "id": "code_15",
        "description": "Implement `memoize(fn)` decorator that caches function results based on arguments. Handle both positional and keyword args.",
        "domain": "coding",
    },
    {
        "id": "code_16",
        "description": "Write a function `diff(old, new)` that computes the minimal edit operations (insert, delete, replace) to transform one string into another.",
        "domain": "coding",
    },
    {
        "id": "code_17",
        "description": "Implement a basic event emitter class with `on(event, callback)`, `emit(event, *args)`, and `off(event, callback)` methods.",
        "domain": "coding",
    },
    {
        "id": "code_18",
        "description": "Fix the memory leak in this long-running process: cached results are never evicted. Add TTL-based expiration.",
        "domain": "coding",
    },
    {
        "id": "code_19",
        "description": "Implement `batch_process(items, batch_size, fn)` that processes items in chunks, collecting results. Handle partial final batches.",
        "domain": "coding",
    },
    {
        "id": "code_20",
        "description": "Write a context manager `@timer` that measures and logs execution time of a code block, with optional label parameter.",
        "domain": "coding",
    },
]

WRITING_TASKS = [
    {
        "id": "write_01",
        "description": "Explain what a hash table is to someone who has never programmed. Use everyday analogies. Keep it under 300 words.",
        "domain": "writing",
    },
    {
        "id": "write_02",
        "description": "Write a step-by-step tutorial for setting up a Python virtual environment and installing packages. Target audience: new developers.",
        "domain": "writing",
    },
    {
        "id": "write_03",
        "description": "Summarize the key ideas of the CAP theorem in distributed systems. Explain the tradeoffs clearly for a technical but non-specialist audience.",
        "domain": "writing",
    },
    {
        "id": "write_04",
        "description": "Write 5 clear, helpful error messages for common form validation failures: empty required field, invalid email, password too short, mismatched passwords, invalid date.",
        "domain": "writing",
    },
    {
        "id": "write_05",
        "description": "Structure an argument for why automated testing is worth the upfront investment. Use the problem-evidence-solution framework.",
        "domain": "writing",
    },
    {
        "id": "write_06",
        "description": "Explain the difference between concurrency and parallelism using a restaurant kitchen analogy. Make it intuitive.",
        "domain": "writing",
    },
    {
        "id": "write_07",
        "description": "Write a tutorial on how to debug a Python program systematically: reproduce, isolate, identify, fix, verify. Include practical tips.",
        "domain": "writing",
    },
    {
        "id": "write_08",
        "description": "Explain why immutability matters in software design. Give three concrete benefits with brief examples for each.",
        "domain": "writing",
    },
    {
        "id": "write_09",
        "description": "Write a clear, jargon-free explanation of how HTTPS keeps web traffic secure. Target: non-technical business stakeholders.",
        "domain": "writing",
    },
    {
        "id": "write_10",
        "description": "Summarize the tradeoffs between SQL and NoSQL databases. Present as a balanced comparison, not advocacy for either.",
        "domain": "writing",
    },
    {
        "id": "write_11",
        "description": "Write a post-mortem template for software incidents. Include sections for timeline, root cause, impact, resolution, and action items.",
        "domain": "writing",
    },
    {
        "id": "write_12",
        "description": "Explain the concept of technical debt to a non-technical project manager. Use the financial debt metaphor but also explain where it breaks down.",
        "domain": "writing",
    },
    {
        "id": "write_13",
        "description": "Write a tutorial on writing good commit messages. Cover the what, why, and conventions. Include good and bad examples.",
        "domain": "writing",
    },
    {
        "id": "write_14",
        "description": "Explain how garbage collection works in managed languages. Use the analogy of a cleaning crew in an office building.",
        "domain": "writing",
    },
    {
        "id": "write_15",
        "description": "Write a clear comparison of REST vs GraphQL APIs. Explain when each is the better choice, with concrete scenarios.",
        "domain": "writing",
    },
    {
        "id": "write_16",
        "description": "Explain the Observer pattern to a junior developer. Start with the real-world analogy of a newspaper subscription.",
        "domain": "writing",
    },
    {
        "id": "write_17",
        "description": "Write an onboarding guide section explaining your team's branching strategy (GitFlow). Cover feature branches, releases, and hotfixes.",
        "domain": "writing",
    },
    {
        "id": "write_18",
        "description": "Explain what Docker containers are and why they're useful. Avoid jargon. Target: a developer who has only used bare metal.",
        "domain": "writing",
    },
    {
        "id": "write_19",
        "description": "Structure a persuasive case for migrating from monolith to microservices. Address both benefits and risks honestly.",
        "domain": "writing",
    },
    {
        "id": "write_20",
        "description": "Write clear release notes for a fictional v2.0 update that adds real-time collaboration, improves search speed 3x, and deprecates the old API.",
        "domain": "writing",
    },
]

DOCUMENTATION_TASKS = [
    {
        "id": "doc_01",
        "description": """Document this Python module. Describe the purpose, explain each function, provide usage examples, and note edge cases.

```python
import hashlib
import json
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir: str = "./config"):
        self._dir = Path(config_dir)
        self._cache = {}
        self._checksums = {}

    def load(self, name: str, default: dict | None = None) -> dict:
        path = self._dir / f"{name}.json"
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"Config '{name}' not found at {path}")
        checksum = hashlib.md5(path.read_bytes()).hexdigest()
        if name in self._cache and self._checksums.get(name) == checksum:
            return self._cache[name]
        with open(path) as f:
            data = json.load(f)
        self._cache[name] = data
        self._checksums[name] = checksum
        return data

    def reload(self, name: str) -> dict:
        self._cache.pop(name, None)
        self._checksums.pop(name, None)
        return self.load(name)

    def list_configs(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json")]
```""",
        "domain": "documentation",
    },
    {
        "id": "doc_02",
        "description": """Write a README for this small CLI project. Include: what it does, installation, usage with examples, and configuration.

The project is called `logwatch` - a command-line tool that monitors log files in real-time, highlights errors in red, warnings in yellow, and lets you filter by regex pattern. It supports multiple files simultaneously and can output to both terminal and a summary file. Written in Python, uses `click` for CLI and `watchdog` for file monitoring.""",
        "domain": "documentation",
    },
    {
        "id": "doc_03",
        "description": """Create an API reference with examples for this HTTP service:

Endpoints:
- POST /tasks - Create a task (body: {title, description, priority, due_date})
- GET /tasks - List tasks (query params: status, priority, page, per_page)
- GET /tasks/:id - Get a single task
- PUT /tasks/:id - Update a task
- DELETE /tasks/:id - Delete a task
- POST /tasks/:id/complete - Mark as complete
- GET /tasks/stats - Get summary statistics

Auth: Bearer token in Authorization header. Rate limit: 100 req/min. Errors return {error: string, code: int}.""",
        "domain": "documentation",
    },
    {
        "id": "doc_04",
        "description": """Write a technical decision record (ADR) explaining why the team chose SQLite over PostgreSQL for the embedded analytics module.

Context: The analytics module runs inside the main application process, tracks user behavior events, and generates daily reports. Data volume is ~10K events/day per instance. The application is deployed as a single binary to customer machines (no cloud). Previously used PostgreSQL but customers complained about the operational overhead of managing a separate database server.""",
        "domain": "documentation",
    },
    {
        "id": "doc_05",
        "description": """Write a changelog entry for v3.2.0 that communicates both what changed and why.

Changes:
1. Replaced the custom connection pool with `sqlx` built-in pool (was causing connection leaks under load)
2. Added request deduplication middleware (users were double-submitting forms)
3. Migrated from `chrono` to `time` crate (chrono has unsound `localtime_r` usage)
4. Bumped minimum supported Rust version to 1.75 (needed for async trait stabilization)
5. Fixed panic when parsing dates with timezone offset > +14:00
6. Deprecated the /v1/sync endpoint (replaced by /v2/stream in v3.0)""",
        "domain": "documentation",
    },
]


def get_training_tasks(domain: str) -> list[dict]:
    if domain == "coding":
        return CODING_TASKS
    elif domain == "writing":
        return WRITING_TASKS
    else:
        raise ValueError(f"Unknown training domain: {domain}")


def get_eval_tasks() -> list[dict]:
    return DOCUMENTATION_TASKS
