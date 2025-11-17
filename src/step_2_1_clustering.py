import os
from collections import defaultdict

import numpy as np
import torch
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from config.utils import Configuration, init_config
from src.utils.io import load_json, save_json
from src.utils.logger import logger


def load_fact_data(input_dir: str) -> tuple[list[dict], list[str]]:
    """Loads all fact data and texts from the fact directory."""
    fact_data = []
    fact_texts = []
    fact_files_name = os.listdir(input_dir)
    logger.info(f"Loading fact data from {len(fact_files_name)} files.")

    for fact_file_name in tqdm(fact_files_name, desc="Loading fact files"):
        data = load_json(os.path.join(input_dir, fact_file_name))
        for extraction in data.get("extractions", []):
            try:
                fact_info = {
                    "id": data["id"],
                    "subject": data["subject"],
                    "source": data["source"],
                    "factual_sentence": extraction["factual_sentence"],
                    "entities": extraction["entities"],
                }
                fact_data.append(fact_info)
                fact_texts.append(extraction["factual_sentence"])
            except:
                logger.error(f"Error: {fact_file_name}")
                continue
    return fact_data, fact_texts


def get_embeddings(texts: list[str], model_path: str, batch_size: int):
    """Generates embeddings for a given list of texts."""
    logger.info(f"Loading embedding model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    logger.info("Starting embedding generation...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().float().numpy()
            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def main():
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    clustering_config = config()["step_2_1_clustering"]
    input_dir = os.path.join(project_root, clustering_config["input_dir"])
    logger.info(f"Input dir: {input_dir}")
    output_dir = os.path.join(project_root, clustering_config["output_dir"])
    logger.info(f"Output dir: {output_dir}")
    batch_size = clustering_config["batch_size"]
    model_path = clustering_config["embedding_model"]
    logger.info(f"Batch size: {batch_size}, Model path: {model_path}")
    dedup_threshold = clustering_config["deduplication_threshold"]
    seed = clustering_config["seed"]
    logger.info(f"Seed: {seed}, Deduplication Threshold: {dedup_threshold}")

    # 1. Load data
    fact_data, fact_texts = load_fact_data(input_dir)

    # 2. Generate vector embeddings
    fact_embeddings = get_embeddings(fact_texts, model_path, batch_size)
    logger.info(f"Embedding generation complete: {fact_embeddings.shape}")

    # 3. Clustering based on BERTopic
    logger.info("Starting clustering with BERTopic...")
    umap_params = clustering_config["umap_params"]
    umap_model = UMAP(random_state=seed, **umap_params)

    hdbscan_params = clustering_config["hdbscan_params"]
    hdbscan_model = HDBSCAN(**hdbscan_params)

    vectorizer_params = clustering_config["vectorizer_params"]
    vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    vectorizer_model = CountVectorizer(**vectorizer_params)

    ctfidf_params = clustering_config["ctfidf_params"]
    ctfidf_model = ClassTfidfTransformer(**ctfidf_params)

    bertopic_params = clustering_config["bertopic_params"]
    topic_model = BERTopic(
        **bertopic_params,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True,
    )

    # Perform clustering by directly passing the embeddings
    topics, probs = topic_model.fit_transform(fact_texts, fact_embeddings)

    # 4. Process and save clustering results
    logger.info("Saving clustering results...")
    clusters = defaultdict(list)
    for fact, topic_id, prob_vector in tqdm(
        zip(fact_data, topics, probs, strict=False), desc="Saving clustering results"
    ):
        fact_info = fact.copy()
        fact_info["assigned_topic_id"] = int(topic_id)
        fact_info["assigned_topic_prob"] = float(prob_vector[topic_id])

        # Add information about the top N topics with the highest probability
        related_topics = []
        for t_idx, prob in enumerate(prob_vector):
            related_topics.append(
                {
                    "topic_id": int(t_idx),
                    "probability": float(prob),
                }
            )
        fact_info["related_topics"] = related_topics
        clusters[topic_id].append(fact_info)
    logger.info(f"Clustering results saved: {len(clusters)} clusters")

    for topic_id, cluster_fact in clusters.items():
        filename = f"cluster_{topic_id}_components.json"
        if topic_id == -1:
            filename = "outliers.json"

        save_path = os.path.join(output_dir, filename)
        save_json(save_path, cluster_fact)
        logger.info(f"Topic {topic_id} ({len(cluster_fact)} facts) saved: {save_path}")

    # Save topic summary information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, "topic_summary.csv"), index=False)
    logger.info(
        f"Topic summary information saved: {os.path.join(output_dir, 'topic_summary.csv')}"
    )


if __name__ == "__main__":
    main()
