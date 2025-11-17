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

    # Load schema-related settings
    extract_fact_config = config()["step_1_extract_fact"]
    input_filepath = extract_fact_config["input_filepath"]
    filepath = os.path.join(project_root, input_filepath)
    logger.info(f"Input filepath: {filepath}")

    output_dir = extract_fact_config["output_dir"]
    output_dir = os.path.join(project_root, output_dir)
    logger.info(f"Output dir: {output_dir}")

    input_data = load_json(filepath)
    logger.info(f"Input data: {input_data[0]}")

    llm_client = create_openai_llm_client(extract_fact_config, config)
    logger.info(f"LLM client: {llm_client}")

    # Extract facts from the input data
    system_prompt = extract_fact_config["system_prompt"]
    for x in input_data:
        save_name = f"fact_{x['id']}.json"
        save_filepath = os.path.join(output_dir, save_name)
        if os.path.exists(save_filepath):
            logger.info(f"Skipping: {save_filepath}")
            continue
        text = x["text"]
        response = llm_client.generate(
            system_prompt=system_prompt,
            chat_messages=[{"role": "user", "content": text}],
        )
        result = parse_json(response.choices[0].message.content)
        result["id"] = x["id"]
        result["subject"] = x["subject"]
        result["source"] = x["source"]
        result["text"] = x["text"]
        save_json(save_filepath, result)
        logger.info(f"Saved: {save_filepath}")


if __name__ == "__main__":
    main()
