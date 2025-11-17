from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class ProjectConfig(BaseSettings):
    """Manages project path related settings.

    Attributes:
        root (str): Project root path
    """

    root: str

    class Config:
        env_prefix = "PROJECT_"


class OpenAIConfig(BaseSettings):
    """Manages OpenAI API settings.

    Attributes:
        api_key (str): OpenAI API key
        embedding_model (str): Embedding model name
    """

    api_key: str
    embedding_model: str

    class Config:
        env_prefix = "OPENAI_"


class OpenRouterConfig(BaseSettings):
    """Manages OpenRouter API settings.

    Attributes:
        api_key (str): OpenRouter API key
    """

    api_key: str

    class Config:
        env_prefix = "OPENROUTER_"


class EnvConfig(BaseSettings):
    """Main settings class that integrates all environment settings.

    Manages settings for each service as a sub-settings class.

    Attributes:
        project (ProjectConfig): Project settings
        openai (OpenAIConfig): OpenAI API settings
        openrouter (OpenRouterConfig): OpenRouter API settings
    """

    project: ProjectConfig = ProjectConfig()
    openai: OpenAIConfig = OpenAIConfig()
    openrouter: OpenRouterConfig = OpenRouterConfig()
