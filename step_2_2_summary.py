import os

import pandas as pd
from tqdm import tqdm

from config.utils import Configuration, init_config
from utils.credential import create_openai_llm_client
from utils.io import load_json, save_json
from utils.logger import logger


def load_cluster_data(input_dir: str) -> tuple[dict[int, list[dict]], list[dict]]:
    """Loads all cluster and fact data from the clustering result directory."""
    topic_info = pd.read_csv(os.path.join(input_dir, "topic_summary.csv"))
    clusters = {}
    for file_path in os.listdir(input_dir):
        if not file_path.endswith("components.json"):
            continue

        fact_data = load_json(os.path.join(input_dir, file_path))
        topic_id = fact_data[0]["assigned_topic_id"]
        topic_info_row = topic_info[topic_info["Topic"] == topic_id]
        clusters[topic_id] = {
            "topic_key": topic_info_row["Name"].values[0],
            "topic_representation": topic_info_row["Representation"].values[0],
            "fact_data": fact_data,
        }
        logger.debug(f"Topic ID: {topic_id} loaded. {len(fact_data)} facts")
    return clusters


def main():
    """Main execution function."""
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    summary_config = config()["step_2_2_summary"]
    input_dir = os.path.join(project_root, summary_config["input_dir"])
    logger.info(f"Input dir: {input_dir}")
    output_dir = os.path.join(project_root, summary_config["output_dir"])
    logger.info(f"Output dir: {output_dir}")

    # 1. Load cluster data
    clusters = load_cluster_data(input_dir)
    if not clusters:
        logger.error("No cluster data found. Please check the results of the previous step.")
        return

    # 2. Summarize clusters
    system_prompt = summary_config["system_prompt"]
    llm_client = create_openai_llm_client(summary_config, config)
    for topic_id, cluster in tqdm(clusters.items(), desc="Generating cluster summaries"):
        topic_representation = cluster["topic_representation"]
        cleaned_facts = [
            {
                "subject": fact["subject"],
                "source": fact["source"],
                "factual_sentence": fact["factual_sentence"],
                "probability": fact["assigned_topic_prob"],
            }
            for fact in cluster["fact_data"]
        ]
        user_prompt = (
            f"# Input:\n{cleaned_facts} \n# Topic Words:\n{topic_representation}"
        )
        response = llm_client.generate(
            system_prompt=system_prompt,
            chat_messages=[{"role": "user", "content": user_prompt}],
        )
        summary_text = response.choices[0].message.content
        cluster["summary"] = summary_text
        save_json(
            os.path.join(output_dir, f"cluster_{topic_id}_summary.json"),
            cluster,
        )
        logger.info(
            f"Saved: {os.path.join(output_dir, f'cluster_{topic_id}_summary.json')}"
        )


if __name__ == "__main__":
    main()
