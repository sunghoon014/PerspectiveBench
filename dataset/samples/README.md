# Input examples

These examples describe a fictional Luma irrigation system and illustrate the construction input and benchmark output formats.

| File | Purpose |
|---|---|
| [chunks.json](chunks.json) | Three source chunks showing the `id`, `subject`, `source`, and `text` input fields. |
| [relation_palette.json](relation_palette.json) | Operation and monitoring relations supplied to the relation-generation prompt. |
| [benchmark.jsonl](benchmark.jsonl) | Two manually assembled items illustrating the final benchmark schema, with different target perspectives over the same valid paths. |

The three chunks can be used for a step 1 extraction trial with API access. Use a larger corpus for the full pipeline. `benchmark.jsonl` is a manually written format example.

Follow the [custom-data guide](../../docs/construction.md) to prepare your corpus, configure isolated stage outputs, and export the final JSONL.

License: [CC BY-NC-SA 4.0](../LICENSE).
