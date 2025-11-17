import os

from config.utils import Configuration, init_config
from utils.credential import create_openai_llm_client
from utils.io import load_json, save_json
from utils.logger import logger
from utils.processor import parse_json


def main():
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    focus_evidence_config = config()["step_4_3_focus_evidence"]
    question_dir = focus_evidence_config["question_dir"]
    question_dir = os.path.join(project_root, question_dir)
    logger.info(f"Question dir: {question_dir}")

    output_dir = focus_evidence_config["output_dir"]
    output_dir = os.path.join(project_root, output_dir)
    logger.info(f"Output dir: {output_dir}")

    system_prompt = focus_evidence_config["system_prompt"]
    llm_client = create_openai_llm_client(focus_evidence_config, config)
    logger.info(f"LLM client: {llm_client}")

    file_list = os.listdir(question_dir)
    for file in file_list:
        question_filepath = os.path.join(question_dir, file)
        question_data = load_json(question_filepath)
        save_name = file.replace(".json", "_focus_evidence.json")
        save_filepath = os.path.join(output_dir, save_name)
        if os.path.exists(save_filepath):
            logger.info(f"Already focused: {save_filepath}")
            continue
        logger.info(f"Save filepath: {save_filepath}")

        # TARGET_REASON
        valid_paths = question_data["valid_paths"]
        controllable_benchmarks = question_data["controllable_questions"]
        controllable_benchmarks = controllable_benchmarks["controllable_benchmarks"]
        perspective_list = []
        for benchmark in controllable_benchmarks:
            perspective_theme = benchmark["theme_name"]
            perspective_target_question = benchmark["focused_question"]
            try:
                valid_paths_for_this_theme = benchmark[
                    "valid_paths_for_this_theme"
                ].copy()
            except:
                valid_paths_for_this_theme = benchmark[
                    "valid_paths_for this_theme"
                ].copy()
            perspective_valid_paths = []
            for path_theme_name in valid_paths_for_this_theme:
                for path in valid_paths:
                    check = path.get(path_theme_name, None)
                    if check:
                        perspective_valid_paths.append(path)
            perspective_list.append(
                {
                    "perspective_theme": perspective_theme,
                    "perspective_target_question": perspective_target_question,
                    "perspective_valid_paths": perspective_valid_paths,
                }
            )

        # EVIDENCE_POOL
        full_content_pool = question_data["full_content_pool"]

        # Focused Evidence 추출 시작
        result_list = []
        for perspective in perspective_list:
            perspective_theme = perspective["perspective_theme"]
            perspective_target_question = perspective["perspective_target_question"]
            perspective_valid_paths = perspective["perspective_valid_paths"]
            user_prompt = f"# PERSPECTIVE_THEME: {perspective_theme}\n\n---\n\n# PERSPECTIVE_TARGET_QUESTION:\n{perspective_target_question}\n\n---\n\n# TARGET_REASON:\n{perspective_valid_paths}\n\n---\n\n# EVIDENCE_POOL:\n{full_content_pool}"
            response = llm_client.generate(
                system_prompt=system_prompt,
                chat_messages=[{"role": "user", "content": user_prompt}],
            )
            focused_evidence = response.choices[0].message.content
            focused_evidence = parse_json(focused_evidence)
            result_dict = {
                "perspective_theme": perspective_theme,
                "perspective_target_question": perspective_target_question,
                "perspective_valid_paths": perspective_valid_paths,
                "focused_evidence": focused_evidence,
                "valid_paths": valid_paths,
                "full_content_pool": full_content_pool,
            }
            result_list.append(result_dict)

        save_json(save_filepath, result_list)
        logger.info(f"Saved: {save_filepath}")


if __name__ == "__main__":
    main()
