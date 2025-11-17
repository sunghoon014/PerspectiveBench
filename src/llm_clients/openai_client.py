import base64
import mimetypes
from pathlib import Path

from openai import APIError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.utils.logger import logger

REASONING_MODELS = ["o1", "o3-mini", "openai/o3-mini", "openai/o4-mini"]


def _log_before_sleep(retry_state):
    """Tenacity callback function to log before retrying."""
    exception = retry_state.outcome.exception()
    logger.warning(
        f"API error occurred: {exception}. "
        f"Retrying in {retry_state.next_action.sleep:.2f} seconds..."
    )


class OpenAILLMClient:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str = None,
        params: dict = None,
        **kwargs,
    ):
        self._model = model
        self._provider = provider
        self._api_key = api_key
        self._params = params
        self._config = kwargs
        self._sync_client = self._create_clients()

    def _create_clients(self) -> OpenAI:
        """Creates a client compatible with the OpenAI library."""
        if self._provider == "openrouter":
            client_config = {
                "api_key": self._api_key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            logger.info("Creating OpenRouter client.")
            return OpenAI(**client_config)
        else:
            client_config = {"api_key": self._api_key}
            logger.info("Creating default OpenAI client.")
            return OpenAI(**client_config)

    def _image_to_data_url(self, image_input: str | bytes) -> str:
        """Converts an image file path, URL, or bytes to a data URL."""
        if isinstance(image_input, bytes):
            # Process byte data
            mime_type = "image/png"  # Default value, can add inference logic if needed
            base64_data = base64.b64encode(image_input).decode("utf-8")
            return f"data:{mime_type};base64,{base64_data}"

        if not isinstance(image_input, str):
            raise TypeError("Unsupported image input type")

        # Return URL or data URL as is
        if image_input.startswith(("http://", "https://", "data:")):
            return image_input

        # Process local file path
        try:
            file_path_str = (
                image_input[7:] if image_input.startswith("file://") else image_input
            )
            path_obj = Path(file_path_str)

            if not path_obj.exists() or not path_obj.is_file():
                logger.warning(f"Local file not found: {file_path_str}")
                return image_input  # Return original if conversion fails

            mime_type, _ = mimetypes.guess_type(str(path_obj))
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"  # Default MIME type

            with open(path_obj, "rb") as f:
                file_data = f.read()

            base64_data = base64.b64encode(file_data).decode("utf-8")
            return f"data:{mime_type};base64,{base64_data}"

        except Exception as e:
            logger.error(f"Failed to convert file {image_input} to data URL: {e}")
            return image_input

    def _validate_and_prepare_messages(
        self,
        system_prompt: str,
        chat_messages: list[dict] | None = None,
        images: list[str | dict | bytes] | None = None,
    ) -> list[dict]:
        """Validate and prepare messages."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_messages:
            messages.extend(chat_messages)

        # Image processing
        if images:
            # Find or create a user message to add the image to.
            if not messages or messages[-1]["role"] != "user":
                messages.append({"role": "user", "content": []})

            last_msg = messages[-1]
            content = last_msg.get("content", "")

            # Convert content to list format.
            if isinstance(content, str):
                content = [{"type": "text", "text": content}] if content else []
            elif not isinstance(content, list):
                content = []

            # Add images
            for img in images:
                if isinstance(img, dict):  # Support existing dict format
                    content.append(img)
                else:  # str (path/URL) or bytes
                    data_url = self._image_to_data_url(img)
                    content.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )

            last_msg["content"] = content

        return messages

    def _prepare_completion_params(
        self, messages: list[dict], stream: bool = False, **kwargs
    ) -> dict:
        """Filters out parameters that are not allowed."""
        allowed_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "n",
            "response_format",
            "tools",
            "tool_choice",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "user",
            # Special parameters for some models
            "reasoning_effort",
            "max_completion_tokens",
        }

        # 1. Start with the base parameters.
        params = self._params.copy()

        # 2. Filter and overwrite with allowed parameters from kwargs.
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
        params.update(filtered_kwargs)

        # 3. Process model-specific special parameters.
        if self._model in REASONING_MODELS:
            params["max_completion_tokens"] = params.pop("max_tokens", 1024)
            params.setdefault("reasoning_effort", "medium")
            params.pop("temperature", None)
        else:
            # Set default values for general models.
            params.setdefault("max_tokens", 1024)
            params.setdefault("temperature", 0.0)

        # 4. Add required parameters.
        params["model"] = self._model
        params["messages"] = messages
        params["stream"] = stream

        return params

    @property
    def model(self) -> str:
        return self._model

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        before_sleep=_log_before_sleep,
    )
    def generate(
        self,
        system_prompt: str,
        chat_messages: list[dict[str, any]] | None = None,
        images: list[str | dict] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        messages = self._validate_and_prepare_messages(
            system_prompt, chat_messages, images
        )

        params = self._prepare_completion_params(messages, stream=False, **kwargs)
        response = self._sync_client.chat.completions.create(**params)
        return response
