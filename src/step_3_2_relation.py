import os
import random

from tqdm import tqdm

from config.utils import Configuration, init_config
from src.utils.credential import create_openai_llm_client
from src.utils.io import load_json, save_json
from src.utils.logger import logger
from src.utils.processor import parse_json

random.seed(104)


def load_summary_data(input_dir: str) -> tuple[dict[int, list[dict]], list[dict]]:
    """Loads all cluster and summary data from the summary directory."""
    clusters = {}
    # Load all cluster result files including outliers.json.
    for file_path in os.listdir(input_dir):
        summary_data = load_json(os.path.join(input_dir, file_path))
        topic_id = int(file_path.split("_")[1])
        summary = summary_data["summary"]
        topic_representation = summary["topic_representation"]
        fact_data = summary["fact_data"]
        fact_data.sort(key=lambda x: x["assigned_topic_prob"], reverse=True)
        fact_text = [x["factual_sentence"] for x in fact_data[:5]]
        clusters[topic_id] = {
            "Summary": summary,
            "Topic Words": topic_representation,
            "Key Evidence": fact_text,
        }
    return clusters


def main():
    """Main execution function."""
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    relation_config = config()["step_3_2_relation"]
    initial_palette_filepath = os.path.join(
        project_root, relation_config["initial_palette"]
    )
    logger.info(f"Initial palette filepath: {initial_palette_filepath}")
    summary_dir = os.path.join(project_root, relation_config["summary_dir"])
    logger.info(f"Summary dir: {summary_dir}")
    candidate_filepath = os.path.join(
        project_root, relation_config["candidate_filepath"]
    )
    logger.info(f"Candidate filepath: {candidate_filepath}")
    output_dir = os.path.join(project_root, relation_config["output_dir"])
    logger.info(f"Output dir: {output_dir}")

    # 1. Load candidate pairs
    candidate_pairs = load_json(candidate_filepath)
    logger.info(f"Candidate pairs count: {len(candidate_pairs)}")

    # 2. Load meta-NCU data
    summary_data = load_summary_data(summary_dir)
    logger.info(f"Summary data count: {len(summary_data)}")

    # 3. Load initial palette
    initial_palette = load_json(initial_palette_filepath)
    logger.info(f"Initial palette: {initial_palette}")

    # 3. Generate relations only for valid pairs.
    system_prompt = relation_config["system_prompt"]
    llm_client = create_openai_llm_client(relation_config, config)
    candidate_pairs_valid = [x for x in candidate_pairs if x["score"] >= 1]
    logger.info(f"Valid candidate pairs count: {len(candidate_pairs_valid)}")
    saved_filepath = output_dir
    relation_generation_results = []
    for candidate_pair in tqdm(candidate_pairs_valid, desc="Relation generation"):
        topic_a_id = candidate_pair["topic_a"]
        topic_b_id = candidate_pair["topic_b"]
        if candidate_pair["score"] < 4:
            continue
        logger.info(f"{topic_a_id}, {topic_b_id} : Score: {candidate_pair['score']}")
        topic_a_data = summary_data[topic_a_id]
        topic_b_data = summary_data[topic_b_id]
        user_prompt = f"# Topic ID: {topic_a_id}\n{topic_a_data}\n\n----\n\n# Topic ID: {topic_b_id}\n{topic_b_data}\n\n----\n\n# Palette:\n{initial_palette}"
        response = llm_client.generate(
            system_prompt=system_prompt,
            chat_messages=[{"role": "user", "content": user_prompt}],
        )
        relation_generation = response.choices[0].message.content
        relation_generation = parse_json(relation_generation)

        candidate_pair["hypotheses"] = relation_generation["hypotheses"]
        relation_generation_results.append(candidate_pair)
        logger.info(f"Done: {topic_a_id}, {topic_b_id}")
        save_json(saved_filepath, relation_generation_results)
        logger.info(f"Saved: {saved_filepath}")


if __name__ == "__main__":
    main()
