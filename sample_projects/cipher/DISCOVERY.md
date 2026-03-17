# Cipher Discovery: SDM Surfaces Code Ecosystems, Not Just Matches

## Date: 2026-02-13

## The Finding

When querying SDM with a code description like "make an HTTP request", the **direct nearest neighbors** (vector DB behavior) return surface-similar snippets: HTTP GET, HTTP POST, file download, Flask server, TCP socket — all "network-y" things.

But the **EAM reconstruction** surfaces a different set: HTTP GET, HTTP POST, then **JSON file reading, JSON file writing, CSV writing** — the patterns that *typically accompany* HTTP requests in real code. You fetch data, then you parse and save it.

The EAM doesn't just find the closest match. It reconstructs the **code ecosystem** around a concept.

## Key Results

### Query: "make an HTTP request"

**Vector DB (direct nearest):**
1. HTTP GET request (0.773)
2. HTTP POST request (0.593)
3. Download a file (0.388)
4. Flask web server (0.353)
5. TCP socket server (0.325)

**SDM Reconstruction nearest:**
1. HTTP GET request (0.778)
2. HTTP POST request (0.754)
3. Read a JSON file (0.725)
4. Write data to JSON file (0.717)
5. Write dicts to CSV file (0.597)

Three snippets surfaced by SDM that direct nearest missed:
- Read a JSON file
- Write data to a JSON file
- Write a list of dicts to CSV

These are exactly what you'd *typically need alongside* HTTP requests — the surrounding data-handling patterns.

### Complete: "import requests"

SDM correctly reconstructs toward the HTTP request cluster, with JSON/CSV data handling bleeding in — the "what typically follows `import requests`" pattern.

### Blend: "database + authentication"

Rather than returning the literal SQLite or bcrypt snippets, the blend surfaced **architectural infrastructure**: pub/sub event systems, Pydantic validation, LRU caches, rate limiters. These are "stateful infrastructure that mediates access" — the conceptual space *between* database and auth.

## Why This Matters

| | Vector DB | SDM (Cipher) |
|---|---|---|
| Query result | Closest single snippets by embedding distance | Reconstructed pattern from ALL similar code |
| What surfaces | Things that LOOK similar | Things that BELONG together |
| "HTTP request" | Network-related code | Network code + data parsing/saving |
| Concept blend | Average of two queries | The architectural intersection |
| Mental model | "Find me the nearest" | "What does this kind of code look like?" |

## The Ecosystem Effect

EAM's superposition means frequently co-occurring patterns reinforce each other. When many snippets involving HTTP requests also involve JSON parsing, those patterns share activation in EAM's hard locations. The reconstruction pulls in the entire ecosystem — not just the query target, but its typical companions.

This is analogous to how human developers think: "I need to make an HTTP request" immediately evokes JSON parsing, error handling, file I/O — the whole workflow, not just `requests.get()`.

## Limitations

With only 30 snippets in 384-dimensional space, the basins are shallow. The blend fidelity was 0.95 (very high), suggesting insufficient interference to form distinct attractor basins. A larger corpus (hundreds of snippets) or lower dimensionality would sharpen the reconstructions and produce more specific intersections.

## Parameters
- Dimension: 384 (sentence-transformers all-MiniLM-L6-v2 default)
- Corpus: 30 Python snippets covering I/O, HTTP, databases, auth, algorithms, infrastructure
- HeatherDB: k=20 activation, Hopfield iterative read, beta=5.0
