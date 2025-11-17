import os

from dependency_injector.providers import Configuration

from config.env_config import EnvConfig


def init_config(config: Configuration) -> None:
    """Initializes the configuration.

    Injects settings from environment variables and YAML files.

    Args:
        config (Configuration): The Configuration object to store the settings in.
    """
    # Inject environment variable settings
    init_env_config(config)

    # Inject yaml settings
    init_yaml_config(config)


def init_yaml_config(config: Configuration) -> None:
    """Loads a YAML configuration file to initialize the settings.

    Loads different configuration files depending on the ENV environment variable,
    and also injects common prompt settings.

    Args:
        config (Configuration): The Configuration object to store the settings in.

    Raises:
        ValueError: If the ENV environment variable is not set.
    """
    if not (env := os.environ.get("ENV")):
        raise ValueError("ENV is not set")

    yaml_path = os.path.join(config.project.root(), "config", f"{env}.yaml")
    prompt_path = os.path.join(config.project.root(), "config", "prompts.yaml")

    # validation
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML Config file not found: {yaml_path}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt Config file not found: {prompt_path}")

    config.from_yaml(yaml_path)
    config.from_yaml(prompt_path)


def init_env_config(config: Configuration) -> None:
    """Loads and initializes settings from environment variables.

    Args:
        config (Configuration): The Configuration object to store the settings in.
    """
    config.from_pydantic(EnvConfig())
