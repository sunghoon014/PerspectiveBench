# Dataset

`dataset/PerspectiveBench.jsonl` contains 1,024 UTF-8 JSON objects, one per question.

| Field | Meaning |
|---|---|
| `id` | Unique integer question identifier |
| `domain` | `physics`, `biology`, or `world_history` |
| `perspective_theme` | Requested explanatory perspective |
| `perspective_target_question` | Question supplied to the responding model |
| `perspective_valid_paths` | Target paths, each with a `path_*` string and a `reason` |
| `valid_paths` | All valid paths, including targets and distractors |
| `focused_evidence` | List of target-perspective source sentences used by Oracle and the paper's main Faithfulness evaluation |
| `full_content_pool` | Graph-node source material, including each node's `Key Evidence` |

A path string has the form `topic_3 -- EXPLAINS_PHENOMENON --> topic_13`. Its number of `-->` edges determines its hop level. Path labels are local to an item. For example, `path_0` in two different items can describe different paths.

The benchmark has no predefined train/validation/test split. Model responses and judge decisions are not part of the dataset.

Source: [OpenStax](https://openstax.org/) textbooks in Physics, Biology, and History.

License: [CC BY-NC-SA 4.0](../dataset/LICENSE). See [data attribution](../dataset/NOTICE.md).
