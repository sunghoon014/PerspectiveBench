import os
import random

from config.utils import Configuration, init_config
from utils.credential import create_openai_llm_client
from utils.io import load_json, save_json
from utils.logger import logger
from utils.processor import parse_json

random.seed(104)


def main():
    config = Configuration()
    init_config(config)

    project_root = config()["project"]["root"]
    logger.info(f"Project root: {project_root}")

    question_generation_config = config()["step_4_2_question_generation"]
    path_filepath = question_generation_config["path_filepath"]
    path_filepath = os.path.join(project_root, path_filepath)
    logger.info(f"Path filepath: {path_filepath}")

    output_dir = question_generation_config["output_dir"]
    output_dir = os.path.join(project_root, output_dir)
    logger.info(f"Output dir: {output_dir}")

    path_data = load_json(path_filepath)
    logger.info(f"Path data: {len(path_data)}")

    llm_client = create_openai_llm_client(question_generation_config, config)
    logger.info(f"LLM client: {llm_client}")

    system_prompt = question_generation_config["system_prompt"]
    logger.info(f"System prompt: {system_prompt}")
    for idx, x in enumerate(path_data):
        save_name = f"question_{idx}.json"
        save_filepath = os.path.join(output_dir, save_name)
        logger.info(f"Save filepath: {save_filepath}")
        if os.path.exists(save_filepath):
            logger.info(f"Already generated: {save_filepath}")
            continue
        path_data = x.copy()
        valid_paths = path_data["valid_paths"]
        del path_data["valid_paths"]
        full_content_pool = path_data
        user_prompt = f"# Valid_Paths_Group:\n{valid_paths}\n\n---\n\n# Full_Content_Pool:\n{full_content_pool}"

        response = llm_client.generate(
            system_prompt=system_prompt,
            chat_messages=[{"role": "user", "content": user_prompt}],
        )
        controllable_questions = response.choices[0].message.content
        controllable_questions = parse_json(controllable_questions)
        result_dict = {
            "controllable_questions": controllable_questions,
            "valid_paths": valid_paths,
            "full_content_pool": full_content_pool,
        }
        save_json(save_filepath, result_dict)
        logger.info(f"Saved: {save_filepath}")


if __name__ == "__main__":
    main()
