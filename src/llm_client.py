from __future__ import annotations

from typing import TypeVar

from ollama import Client
from pydantic import BaseModel, ValidationError

from src.config import settings


ModelType = TypeVar(
    "ModelType",
    bound=BaseModel,
)


GROUNDING_RULES = """
You are part of a data intelligence and data quality system.

Follow these rules strictly:

1. Use only information explicitly provided in the prompt or returned
   by analytical tools.

2. Never invent statistics, values, row counts, distributions,
   business rules, schema information, meanings, relationships,
   or database facts.

3. Never treat an assumption as observed evidence.

4. If evidence is insufficient, say that the information is unknown
   or request an appropriate analytical tool.

5. A hypothesis must clearly be marked as a hypothesis and must be
   supported by actual evidence provided to you.

6. Do not calculate statistics mentally when an analytical tool can
   calculate them exactly.

7. Do not claim a data-quality violation unless evidence supports it.

8. Do not infer business rules as facts. AI-generated rules are
   suggestions unless explicitly supplied as business requirements.

9. Confidence represents confidence in an inference from available
   evidence. It must not be used to hide missing evidence.

10. When structured output is requested, return only content that
    follows the requested schema.

Accuracy is more important than producing more findings.
"""


class LLMError(RuntimeError):
    """
    Raised when the local language model cannot return a usable response.
    """


class LocalLLMClient:
    """
    Small wrapper around Ollama.

    All local-model calls pass through this class so model configuration,
    grounding rules and error handling stay consistent.
    """

    def __init__(self) -> None:
        # Prevent a model request from blocking the whole pipeline
        # indefinitely.
        self.client = Client(
            host=settings.ollama_host,
            timeout=180.0,
        )

        self.model = (
            settings.ollama_model
        )

        # Keep the local model loaded between nearby pipeline stages.
        self.keep_alive = "15m"

        # qwen3:4b is currently running with a 4096-token context in
        # the local Ollama setup.
        self.context_size = 4096

    def _build_system_prompt(
        self,
        system_prompt: str,
    ) -> str:
        """
        Add shared grounding rules to task-specific instructions.
        """

        task_prompt = (
            system_prompt.strip()
            if system_prompt
            else "Follow the grounding rules."
        )

        return (
            f"{GROUNDING_RULES.strip()}\n\n"
            "TASK-SPECIFIC INSTRUCTIONS:\n"
            f"{task_prompt}"
        )

    def check_connection(self) -> bool:
        """
        Check whether the local Ollama service is reachable.
        """

        try:
            self.client.list()
            return True

        except Exception:
            return False

    def check_model(self) -> bool:
        """
        Check whether the configured model is installed locally.
        """

        try:
            response = (
                self.client.list()
            )

        except Exception:
            return False

        models = getattr(
            response,
            "models",
            [],
        )

        for model in models:
            model_name = getattr(
                model,
                "model",
                "",
            )

            if not model_name:
                model_name = getattr(
                    model,
                    "name",
                    "",
                )

            if not model_name:
                continue

            if (
                model_name
                == self.model
            ):
                return True

            # Supports a model configured without a tag, for example
            # "qwen3", when Ollama reports "qwen3:4b".
            if (
                ":" not in self.model
                and model_name.startswith(
                    f"{self.model}:"
                )
            ):
                return True

        return False

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Get a grounded plain-text response from the local model.
        """

        final_system_prompt = (
            self._build_system_prompt(
                system_prompt
            )
        )

        try:
            response = (
                self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                final_system_prompt
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                user_prompt.strip()
                            ),
                        },
                    ],

                    # Long internal reasoning is unnecessary for our
                    # controlled data-pipeline tasks.
                    think=False,

                    keep_alive=(
                        self.keep_alive
                    ),

                    options={
                        "temperature": 0,
                        "seed": 42,
                        "num_ctx": (
                            self.context_size
                        ),
                        "num_predict": 512,
                    },
                )
            )

        except Exception as exc:
            raise LLMError(
                "Unable to communicate with "
                f"Ollama model '{self.model}': "
                f"{exc}"
            ) from exc

        content = getattr(
            response.message,
            "content",
            "",
        )

        if not content:
            raise LLMError(
                "The local model returned an empty response."
            )

        return (
            content.strip()
        )

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelType],
        retries: int = 2,
        num_predict: int = 800,
    ) -> ModelType:
        """
        Request JSON-schema-constrained output and validate it locally
        with Pydantic.

        `num_predict` is configurable because different pipeline stages
        need different response sizes. Small tasks such as rule proposal
        generation should not be allowed to produce unnecessarily large
        outputs.
        """

        if num_predict < 64:
            raise ValueError(
                "num_predict must be at least 64."
            )

        if num_predict > 2048:
            raise ValueError(
                "num_predict must not exceed 2048."
            )

        schema = (
            response_model
            .model_json_schema()
        )

        final_system_prompt = (
            self._build_system_prompt(
                system_prompt
            )
        )

        base_user_prompt = (
            f"{user_prompt.strip()}\n\n"
            "Return only the requested structured result. "
            "Use only the supplied evidence. "
            "If information cannot be determined, keep the result "
            "conservative instead of inventing missing information."
        )

        current_user_prompt = (
            base_user_prompt
        )

        last_error: Exception | None = (
            None
        )

        total_attempts = max(
            1,
            retries + 1,
        )

        for attempt in range(
            total_attempts
        ):
            try:
                response = (
                    self.client.chat(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    final_system_prompt
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    current_user_prompt
                                ),
                            },
                        ],

                        # Ollama constrains output directly using the
                        # Pydantic-derived JSON schema.
                        format=schema,

                        # Disable Qwen3's long reasoning mode.
                        think=False,

                        keep_alive=(
                            self.keep_alive
                        ),

                        options={
                            "temperature": 0,
                            "seed": 42,
                            "num_ctx": (
                                self.context_size
                            ),
                            "num_predict": (
                                num_predict
                            ),
                        },
                    )
                )

                content = getattr(
                    response.message,
                    "content",
                    "",
                )

                if not content:
                    raise LLMError(
                        "The model returned an empty "
                        "structured response."
                    )

                return (
                    response_model
                    .model_validate_json(
                        content
                    )
                )

            except ValidationError as exc:
                last_error = exc

                if (
                    attempt
                    >= total_attempts - 1
                ):
                    break

                # Retry only when the generated JSON does not satisfy
                # the expected Pydantic structure.
                current_user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "The previous result failed structural validation. "
                    "Correct the structure and return it again. "
                    "Do not add explanations or unsupported facts."
                )

            except LLMError:
                raise

            except Exception as exc:
                # Server errors and timeouts are unlikely to improve by
                # immediately repeating the exact same expensive call.
                raise LLMError(
                    "Ollama failed while generating "
                    f"structured output with '{self.model}': "
                    f"{exc}"
                ) from exc

        raise LLMError(
            "The local model failed to produce "
            "a valid structured response after "
            f"{total_attempts} attempt(s). "
            f"Last validation error: {last_error}"
        )


llm_client = LocalLLMClient()