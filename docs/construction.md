# Build from your own documents

Build a benchmark from your own documents by preparing text chunks and a relation palette, running the construction stages, and exporting the reviewed questions as JSONL. Run all commands from the repository root.

The [samples](../dataset/samples/README.md) illustrate the input and output formats using a fictional irrigation system. Replace them with your own corpus before running the full pipeline.

## Prepare document chunks

Create a UTF-8 **JSON array**, following [`dataset/samples/chunks.json`](../dataset/samples/chunks.json):

```json
[
  {
    "id": 0,
    "subject": "demo",
    "source": "luma_manual.txt#operation",
    "text": "In the fictional Luma irrigation system, the Luma controller opens the Luma gate by sending an open command. The Luma gate permits water flow when open. The Luma gate stops water flow when closed."
  }
]
```

| Field | What to provide |
|---|---|
| `id` | A unique integer per chunk. Step 1 uses it in `fact_<id>.json`; reused IDs can cause chunks to be skipped. |
| `subject` | A nonempty subject label for provenance. Process one corpus/domain at a time; this field does not partition clustering automatically. |
| `source` | A document/section identifier. The pipeline copies this value; it does not open the named file or download a URL. |
| `text` | The actual passage supplied to the extraction model. |

Convert PDF, HTML, or other formats into text before this step. Split at coherent paragraph or section boundaries, preserve context needed to resolve references, and remove headers and broken extraction artifacts. Step 1 asks for 3–7 important facts per chunk; choose passages with enough substantive content. The pipeline accepts prepared text chunks, so document conversion and chunking come first.

Keep source titles, authors, URLs, editions, license notices, and modifications alongside your corpus.

## Prepare a relation palette

Start with [`dataset/samples/relation_palette.json`](../dataset/samples/relation_palette.json). It groups proposed relations by explanatory category. Each relation has a `name`, a `description` defining its direction, and an `example` such as `Luma controller -- OPENS --> Luma gate`.

Replace these with relations appropriate to your material, using consistent labels. The palette is passed to the relation model as prompt content, not enforced as a closed schema. The prompt permits new relations; review generated edges and their supporting evidence before treating them as annotations.

## Configure a separate run

```bash
uv sync --locked --extra construction
test -f .env || cp env.sample .env
mkdir -p dataset/custom/my_domain
cp dataset/samples/chunks.json dataset/custom/my_domain/chunks.json
cp dataset/samples/relation_palette.json dataset/custom/my_domain/relation_palette.json
test -f config/custom.yaml || cp config/local.yaml config/custom.yaml
```

Replace the copied JSON examples with your own corpus and palette. `dataset/custom/`, `dataset/processed/`, and `config/custom.yaml` are ignored by Git. Set `OPENROUTER_API_KEY` in `.env` for the default provider; only the selected provider needs a credential. Construction uses paid LLM calls, and F2LLM-4B embedding downloads weights and needs sufficient memory.

Edit `config/custom.yaml`:

- Set `step_1_extract_fact.input_filepath` to `dataset/custom/my_domain/chunks.json`.
- Set `step_3_2_relation.initial_palette` to `dataset/custom/my_domain/relation_palette.json`.
- Replace **every** `dataset/processed/` prefix with `dataset/processed/my_domain/`, including intermediate input paths. Use a new prefix for a new corpus or rerun with changed inputs. Several stages skip existing output files.
- Keep the producer/consumer paths matched as shown below. Despite its name, `step_3_2_relation.output_dir` is a **JSON file path**.

| Stage | Reads | Writes under the selected output prefix |
|---|---|---|
| `step_1_extract_fact` | Your chunk array | `step1_extract_fact/fact_<id>.json` |
| `step_2_1_clustering` | Extracted facts | `step2_clustering/` |
| `step_2_2_summary` | Clusters | `step2_summary/` |
| `step_3_1_filtering` | Clusters | `step3_filtering/candidate_pairs.json` |
| `step_3_2_relation` | Candidates, summaries, palette | `step3_relation/relation_generation.json` |
| `step_4_1_path_generation` | Relations and summaries | `step4_path_generation/path_generation.json` |
| `step_4_2_question_generation` | Path groups | `step4_question_generation/` |
| `step_4_3_focus_evidence` | Questions and path groups | `step4_focus_evidence/` |

The default clustering uses UMAP with 15 neighbors and 5 components, and HDBSCAN with a minimum cluster size of 10. Adjust these settings to your corpus and inspect the clusters before proceeding. Small inputs may fail clustering or produce no usable graph.

## Run the construction stages

Run from the repository root. `ENV=custom` selects `config/custom.yaml`; prompts are merged from `config/prompts.yaml`. The repository root is inferred unless `PROJECT_ROOT` overrides it.

```bash
export ENV=custom
uv run --extra construction python -m src.construction.step_1_extract_fact
uv run --extra construction python -m src.construction.step_2_1_clustering
uv run --extra construction python -m src.construction.step_2_2_summary
uv run --extra construction python -m src.construction.step_3_1_filtering
uv run --extra construction python -m src.construction.step_3_2_relation
uv run --extra construction python -m src.construction.step_4_1_path_generation
uv run --extra construction python -m src.construction.step_4_2_question_generation
uv run --extra construction python -m src.construction.step_4_3_focus_evidence
```

Check each stage's outputs before starting the next. Fresh API runs can produce different facts, relations, and questions.

## Export benchmark JSONL

Step 4.3 writes a **JSON array per question group**. Export one question per JSONL line, adding a unique integer `id` and a `domain` label to each item. See the [dataset schema](dataset.md) for the fields and the [benchmark example](../dataset/samples/benchmark.jsonl) for a complete item.

Before exporting, review the questions, paths, and evidence. Target paths must be a nonempty subset of all valid paths. Off-perspective paths must also be supported by the source; for an exclusion test, include at least one valid path outside the target set.

Save the following as `export_benchmark.py` in the repository root. Set `domain` to your corpus label. The script checks that required fields are present, adds IDs, and writes JSONL without overwriting an existing file.

```python
import json
from pathlib import Path

domain = "my_domain"
source = Path("dataset/processed") / domain / "step4_focus_evidence"
output = Path("dataset/custom") / domain / "benchmark.jsonl"
required = {
    "perspective_theme",
    "perspective_target_question",
    "perspective_valid_paths",
    "valid_paths",
    "focused_evidence",
    "full_content_pool",
}

items = []
for path in sorted(source.glob("*_focus_evidence.json")):
    group = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(group, list):
        raise ValueError(f"Expected a question-group array: {path}")
    for item in group:
        if not isinstance(item, dict):
            raise ValueError(f"Expected question objects: {path}")
        if not required <= item.keys():
            raise ValueError(f"Missing fields in {path}: {required - item.keys()}")
        items.append({**item, "id": len(items), "domain": domain})

if not items:
    raise ValueError("No constructed questions found")
text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as stream:
    stream.write(text)
print(f"Exported {len(items)} items to {output}")
```

Run it from the repository root:

```bash
python3 export_benchmark.py
```

For multiple domains, repeat construction separately and merge the reviewed exports with globally unique IDs before evaluation. Keep the ID mapping once model responses have been generated.
