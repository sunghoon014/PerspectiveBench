import bisect
import os
from itertools import combinations

from tqdm import tqdm

from config.utils import Configuration, init_config
from src.utils.io import load_json, save_json
from src.utils.logger import logger


def load_cluster_data(input_dir: str) -> tuple[dict[int, list[dict]], list[dict]]:
    """Loads all cluster and fact data from the clustering result directory."""
    clusters = {}
    for file_path in os.listdir(input_dir):
        if not file_path.endswith("components.json"):
            continue

        fact_data = load_json(os.path.join(input_dir, file_path))
        topic_id = fact_data[0]["assigned_topic_id"]
        clusters[topic_id] = fact_data
        logger.debug(f"Topic ID: {topic_id} loaded. {len(fact_data)} facts")
    return clusters


def generate_candidate_pairs(clusters: dict[int, list[dict]]) -> list[dict]:
    """Generates and scores candidate cluster pairs."""
    topic_ids = sorted(clusters.keys())
    candidate_pairs = []
    for topic_a, topic_b in tqdm(
        combinations(topic_ids, 2), desc="Generating and scoring candidate pairs"
    ):
        logger.info(f"topic_a: {topic_a}, topic_b: {topic_b}")
        score = 0
        reason = []
        clusters_a = clusters[topic_a]
        clusters_b = clusters[topic_b]

        # 1. Shared entity score
        cluster_a_entities = {
            entity for fact in clusters_a for entity in fact["entities"]
        }
        logger.info(f"cluster_a_entities: {cluster_a_entities}")
        cluster_b_entities = {
            entity for fact in clusters_b for entity in fact["entities"]
        }
        logger.info(f"cluster_b_entities: {cluster_b_entities}")
        shared_entities = cluster_a_entities.intersection(cluster_b_entities)
        logger.info(f"shared_entities: {shared_entities}")
        if len(shared_entities) >= 3:
            score += 5
            reason.append("shared_entities")
        elif 1 <= len(shared_entities) <= 2:
            score += 4
            reason.append("shared_entities")

        # 2. Proximity of original documents
        ids_a = sorted([fact["id"] for fact in clusters_a])
        logger.info(f"ids_a: {ids_a}")
        ids_b = sorted([fact["id"] for fact in clusters_b])
        logger.info(f"ids_b: {ids_b}")
        id_proximity_count = 0
        for id_a in ids_a:
            left_bound = id_a - 5
            right_bound = id_a + 5
            start_index = bisect.bisect_left(ids_b, left_bound)
            end_index = bisect.bisect_right(ids_b, right_bound)
            id_proximity_count += end_index - start_index
        logger.info(f"id_proximity_count: {id_proximity_count}")
        if id_proximity_count >= 5:
            score += 3
            reason.append("id_proximity")
        elif 2 <= id_proximity_count <= 4:
            score += 2
            reason.append("id_proximity")
        elif 0 < id_proximity_count < 2:
            score += 1
            reason.append("id_proximity")

        # 3. BERTopic bridge
        cluster_a_bridge_topics = [
            True
            for fact in clusters_a
            if fact["related_topics"][topic_b]["probability"] > 0.05
        ]
        cluster_b_bridge_topics = [
            True
            for fact in clusters_b
            if fact["related_topics"][topic_a]["probability"] > 0.05
        ]
        logger.info(f"cluster_a_bridge_topics: {cluster_a_bridge_topics}")
        logger.info(f"cluster_b_bridge_topics: {cluster_b_bridge_topics}")
        if (len(cluster_a_bridge_topics) >= 3) and (len(cluster_b_bridge_topics) >= 3):
            score += 1
            reason.append("bertopic_bridge")

        candidate_pairs.append(
            {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "score": score,
                "reason": reason,
            }
        )

    logger.info(f"Generated a total of {len(candidate_pairs)} candidate pairs.")
    return sorted(candidate_pairs, key=lambda x: x["score"], reverse=True)


def main():
    """Main execution function."""
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    filtering_config = config()["step_3_1_filtering"]
    input_dir = os.path.join(project_root, filtering_config["input_dir"])
    logger.info(f"Input dir: {input_dir}")
    output_dir = os.path.join(project_root, filtering_config["output_dir"])
    logger.info(f"Output dir: {output_dir}")

    # 1. Load cluster data
    clusters = load_cluster_data(input_dir)
    if not clusters:
        logger.error(
            "No cluster data found. Please check the results of the previous step."
        )
        return

    # 3. Generate candidate pairs
    candidate_pairs = generate_candidate_pairs(clusters)

    candidates_path = os.path.join(output_dir, "candidate_pairs.json")
    save_json(candidates_path, candidate_pairs)
    logger.info(f"Candidate pairs saved: {candidates_path}")


if __name__ == "__main__":
    main()
