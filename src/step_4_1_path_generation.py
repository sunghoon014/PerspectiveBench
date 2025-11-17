import os
import random

import networkx as nx
import numpy as np

from config.utils import Configuration, init_config
from src.utils.io import load_json, save_json
from src.utils.logger import logger

random.seed(104)


def get_relation_reason(s_hypo):
    """Extracts the relation and reason from a hypothesis."""
    if s_hypo.get("new_relation"):
        relation = s_hypo.get("new_relation")
        if isinstance(relation, dict):
            relation = relation.get("new_relation_name")
    else:
        relation = s_hypo.get("relation")
        if isinstance(relation, dict):
            relation = relation.get("new_relation_name")
    reason = s_hypo.get("reason")
    return relation, reason


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
    return summary_data


def main():
    """Main execution function."""
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    path_config = config()["step_4_1_path_generation"]
    summary_dir = path_config["summary_dir"]
    summary_dir = os.path.join(project_root, summary_dir)
    logger.info(f"Summary dir: {summary_dir}")

    relation_filepath = path_config["relation_filepath"]
    relation_filepath = os.path.join(project_root, relation_filepath)
    logger.info(f"Relation filepath: {relation_filepath}")

    output_filepath = path_config["output_filepath"]
    output_filepath = os.path.join(project_root, output_filepath)
    logger.info(f"Output filepath: {output_filepath}")

    # 1. Load meta-NCU data
    summary_data = load_summary_data(summary_dir)
    logger.info(f"Summary data count: {len(summary_data)}")

    # 2. Load relation data
    relation_data = load_json(relation_filepath)
    logger.info(f"Relation data count: {len(relation_data)}")

    # 3. Generate paths
    result = []
    # Create a graph
    g = nx.DiGraph()
    all_hypotheses_list = []
    for item in relation_data:
        hypotheses = item.get("hypotheses", [])
        all_hypotheses_list.extend(hypotheses)
        for hypo in hypotheses:
            source_topic_id = hypo.get("source_topic_id")
            target_topic_id = hypo.get("target_topic_id")
            score = item.get("score")
            if hypo.get("new_relation"):
                relation = hypo.get("new_relation")
            else:
                relation = hypo.get("relation")
            g.add_edge(
                source_topic_id,
                target_topic_id,
                relation=relation,
                score=score,
            )
    logger.info(f"Graph: {g}")
    logger.info(f"All hypotheses list Sample: {all_hypotheses_list[:2]}")
    logger.info(f"All hypotheses list length: {len(all_hypotheses_list)}")

    # Hop 1 Path Generation
    paths_1_hop = [
        {
            "source_topic_id": u,
            "relation": data["relation"],
            "target_topic_id": v,
            "score": data["score"],
        }
        for u, v, data in g.edges(data=True)
    ]
    paths_1_hop.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Paths 1 hop length: {len(paths_1_hop)}")
    h_c = 0
    for path in paths_1_hop:
        source_topic_id = path["source_topic_id"]
        target_topic_id = path["target_topic_id"]
        selected_hypotheses = [
            x
            for x in all_hypotheses_list
            if x["source_topic_id"] == source_topic_id
            and x["target_topic_id"] == target_topic_id
        ]
        if len(selected_hypotheses) < 3:
            continue
        h_c += 1
        selected_dict = {"valid_paths": []}
        logger.info(f"{source_topic_id}, {target_topic_id}")
        for idx, s_hypo in enumerate(selected_hypotheses):
            relation, reason = get_relation_reason(s_hypo)
            selected_dict["valid_paths"].append(
                {
                    f"path_{idx}": f"topic_{s_hypo['source_topic_id']} -- {relation} --> topic_{s_hypo['target_topic_id']}",
                    "reason": reason,
                }
            )
        source_summary = summary_data[source_topic_id]
        target_summary = summary_data[target_topic_id]
        selected_dict[f"topic_{source_topic_id}_source"] = source_summary
        selected_dict[f"topic_{target_topic_id}_target"] = target_summary
        result.append(selected_dict)

    # Hop 2 Path Generation
    paths_2_hop = []
    for n1 in g.nodes():
        for n2 in g.successors(n1):
            for n3 in g.successors(n2):
                if n3 != n1:
                    path = [n1, n2, n3]
                    score = g[n1][n2]["score"] + g[n2][n3]["score"]
                    paths_2_hop.append(
                        {
                            "path": path,
                            "score": score,
                        }
                    )
    paths_2_hop.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Paths 2 hop length: {len(paths_2_hop)}")
    logger.info(f"Paths 2 hop Sample: {paths_2_hop[:5]}")

    h_c2 = 0
    for path in paths_2_hop:
        topic_id_1 = path["path"][0]
        topic_id_2 = path["path"][1]
        topic_id_3 = path["path"][2]
        selected_hypotheses_1_2 = [
            x
            for x in all_hypotheses_list
            if x["source_topic_id"] == topic_id_1 and x["target_topic_id"] == topic_id_2
        ]
        selected_hypotheses_2_3 = [
            x
            for x in all_hypotheses_list
            if x["source_topic_id"] == topic_id_2 and x["target_topic_id"] == topic_id_3
        ]

        selected_dict = {"valid_paths": []}
        count = 0
        for s_hypo_1_2 in selected_hypotheses_1_2:
            relation_1_2, reason_1_2 = get_relation_reason(s_hypo_1_2)

            for s_hypo_2_3 in selected_hypotheses_2_3:
                relation_2_3, reason_2_3 = get_relation_reason(s_hypo_2_3)

                selected_dict["valid_paths"].append(
                    {
                        f"path_{count}": f"topic_{topic_id_1} -- {relation_1_2} --> topic_{topic_id_2} -- {relation_2_3} --> topic_{topic_id_3}",
                        f"{topic_id_1}_reason_{topic_id_2}": reason_1_2,
                        f"{topic_id_2}_reason_{topic_id_3}": reason_2_3,
                    }
                )
                count += 1
        if not 3 < len(selected_dict["valid_paths"]) < 15:
            continue
        h_c2 += 1

        # Add source information for all related topics.
        selected_dict[f"topic_{topic_id_1}_source"] = summary_data[topic_id_1]
        selected_dict[f"topic_{topic_id_2}_source"] = summary_data[topic_id_2]
        selected_dict[f"topic_{topic_id_3}_source"] = summary_data[topic_id_3]

        result.append(selected_dict)
    logger.info(f"Hop 2 Result length: {h_c2}")
    # Hop 3 Path Generation
    paths_3_hop = []
    for n1 in g.nodes():
        for n2 in g.successors(n1):
            for n3 in g.successors(n2):
                if n3 == n1:
                    continue
                for n4 in g.successors(n3):
                    if n4 in (n1, n2):
                        continue
                    path = [n1, n2, n3, n4]
                    score = g[n1][n2]["score"] + g[n2][n3]["score"] + g[n3][n4]["score"]
                    paths_3_hop.append({"path": path, "score": score})

    paths_3_hop.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Paths 3 hop length: {len(paths_3_hop)}")
    logger.info(f"Paths 3 hop Sample: {paths_3_hop[:5]}")

    h_c3 = 0
    for path in paths_3_hop:
        topic_id_1 = path["path"][0]
        topic_id_2 = path["path"][1]
        topic_id_3 = path["path"][2]
        topic_id_4 = path["path"][3]

        selected_hypotheses_1_2 = [
            x
            for x in all_hypotheses_list
            if x["source_topic_id"] == topic_id_1 and x["target_topic_id"] == topic_id_2
        ]
        selected_hypotheses_2_3 = [
            x
            for x in all_hypotheses_list
            if x["source_topic_id"] == topic_id_2 and x["target_topic_id"] == topic_id_3
        ]
        selected_hypotheses_3_4 = [
            x
            for x in all_hypotheses_list
            if x["source_topic_id"] == topic_id_3 and x["target_topic_id"] == topic_id_4
        ]

        selected_dict = {"valid_paths": []}
        count = 0
        for s_hypo_1_2 in selected_hypotheses_1_2:
            relation_1_2, reason_1_2 = get_relation_reason(s_hypo_1_2)

            for s_hypo_2_3 in selected_hypotheses_2_3:
                relation_2_3, reason_2_3 = get_relation_reason(s_hypo_2_3)

                for s_hypo_3_4 in selected_hypotheses_3_4:
                    relation_3_4, reason_3_4 = get_relation_reason(s_hypo_3_4)

                    selected_dict["valid_paths"].append(
                        {
                            f"path_{count}": f"topic_{topic_id_1} -- {relation_1_2} --> topic_{topic_id_2} -- {relation_2_3} --> topic_{topic_id_3} -- {relation_3_4} --> topic_{topic_id_4}",
                            f"{topic_id_1}_reason_{topic_id_2}": reason_1_2,
                            f"{topic_id_2}_reason_{topic_id_3}": reason_2_3,
                            f"{topic_id_3}_reason_{topic_id_4}": reason_3_4,
                        }
                    )
                    count += 1

        if not 3 < len(selected_dict["valid_paths"]) < 15:
            continue
        h_c3 += 1

        # Add source information for all related topics.
        selected_dict[f"topic_{topic_id_1}_source"] = summary_data[topic_id_1]
        selected_dict[f"topic_{topic_id_2}_source"] = summary_data[topic_id_2]
        selected_dict[f"topic_{topic_id_3}_source"] = summary_data[topic_id_3]
        selected_dict[f"topic_{topic_id_4}_source"] = summary_data[topic_id_4]

        result.append(selected_dict)
    logger.info(f"Hop 3 Result length: {h_c3}")
    logger.info(f"Result length: {len(result)}")
    save_json(output_filepath, result)


if __name__ == "__main__":
    main()
