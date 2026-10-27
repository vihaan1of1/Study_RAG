"""StudyRAG LLM Client Module.

Provides a unified interface supporting Google Gemini (gemini-3.8-flash) and Groq
(llama-3.3-70b-versatile), with structured JSON output parsing, automatic schema enforcement,
retry logic, and graceful error messages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_PROVIDER,
    get_gemini_api_key,
    get_groq_api_key,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM Client supporting Google Gemini and Groq with offline capability."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: float = 45.0,
    ):
        self.provider = (provider or DEFAULT_LLM_PROVIDER or "gemini").strip().lower()
        self.timeout = timeout_seconds

        # Configure provider-specific attributes
        if self.provider == "gemini":
            self.api_key = get_gemini_api_key(api_key)
            self.default_model = default_model or DEFAULT_GEMINI_MODEL
            self._gemini_client = None
            if self.is_configured:
                try:
                    from google import genai
                    self._gemini_client = genai.Client(api_key=self.api_key)
                except Exception as exc:
                    logger.warning(f"Failed to initialize Google GenAI client: {exc}")
        elif self.provider == "groq":
            self.api_key = get_groq_api_key(api_key)
            self.default_model = default_model or DEFAULT_GROQ_MODEL
            self._groq_client = None
            if self.is_configured:
                try:
                    from groq import Groq
                    self._groq_client = Groq(api_key=self.api_key, timeout=self.timeout)
                except Exception as exc:
                    logger.warning(f"Failed to initialize Groq client: {exc}")
        else:
            # Offline or custom fallback
            self.provider = "offline"
            self.api_key = ""
            self.default_model = "local-extractive"

    @property
    def is_configured(self) -> bool:
        """Check if client has an active, valid API key for the selected provider."""
        if self.provider == "gemini":
            return bool(
                self.api_key
                and self.api_key != "your_gemini_api_key_here"
                and len(self.api_key) > 10
            )
        if self.provider == "groq":
            return bool(
                self.api_key
                and self.api_key != "your_groq_api_key_here"
                and len(self.api_key) > 10
            )
        return False

    def _ensure_configured(self) -> None:
        """Raise an informative error if the selected provider lacks an API key."""
        if not self.is_configured:
            if self.provider == "gemini":
                raise ValueError(
                    "Google Gemini API key is not configured. Please add GEMINI_API_KEY to your "
                    ".env file or enter your Gemini key in the StudyRAG sidebar settings."
                )
            elif self.provider == "groq":
                raise ValueError(
                    "Groq API key is not configured. Please add GROQ_API_KEY to your "
                    ".env file or enter your Groq key in the StudyRAG sidebar settings."
                )
            else:
                raise ValueError(
                    "Offline mode active. No LLM provider is connected. Switch to Google Gemini or Groq in the sidebar."
                )

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate conversational response from the configured LLM provider."""
        self._ensure_configured()
        target_model = model or self.default_model

        if self.provider == "gemini":
            return self._generate_gemini_completion(
                messages=messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif self.provider == "groq":
            return self._generate_groq_completion(
                messages=messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise RuntimeError("Cannot generate LLM response in offline mode.")

    def _generate_gemini_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Execute chat completion via Google GenAI SDK."""
        from google import genai
        from google.genai import errors, types

        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)

        # Separate system messages from conversational turns
        system_prompts: List[str] = []
        user_turns: List[types.Content] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            if not content:
                continue

            if role == "system":
                system_prompts.append(content)
            else:
                # Map standard roles to Gemini roles ('user' or 'model')
                gemini_role = "model" if role in ("assistant", "model") else "user"
                user_turns.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        sys_instruction = "\n\n".join(system_prompts) if system_prompts else None

        # Build contents parameter
        if len(user_turns) == 1 and user_turns[0].role == "user":
            contents: Any = user_turns[0].parts[0].text
        elif len(user_turns) > 0:
            contents = user_turns
        else:
            contents = "Hello"

        config = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            response = self._gemini_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return (response.text or "").strip()

        except errors.ClientError as exc:
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Gemini Model '{model}' not found or unavailable. Please use 'gemini-3.8-flash' or 'gemini-3.5-flash-lite'."
                )
            if "401" in msg or "403" in msg or "invalid" in msg.lower():
                raise RuntimeError(
                    "Gemini API key is invalid or unauthorized. Please verify your GEMINI_API_KEY."
                )
            if "429" in msg or "quota" in msg.lower():
                raise RuntimeError(
                    f"Gemini API Rate Limit or Quota Reached: {msg}. Please wait a moment and try again."
                )
            raise RuntimeError(f"Gemini API Client Error: {msg}")

        except errors.ServerError as exc:
            raise RuntimeError(f"Google Gemini Server Error: {str(exc)}. Please try again in a few moments.")
        except Exception as exc:
            raise RuntimeError(f"Unexpected error communicating with Gemini API: {str(exc)}")

    def _generate_groq_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Execute chat completion via Groq API."""
        from groq import APIConnectionError, APIError, AuthenticationError, RateLimitError

        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=self.api_key, timeout=self.timeout)

        try:
            response = self._groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()

        except AuthenticationError:
            raise RuntimeError(
                "Groq Authentication Failed: Invalid API key. Please verify your GROQ_API_KEY."
            )
        except RateLimitError as exc:
            raise RuntimeError(
                f"Groq API Rate Limit Reached: {str(exc)}. Please wait a moment and try again."
            )
        except APIConnectionError:
            raise RuntimeError(
                "Groq Network Connection Error: Could not reach the Groq API server. "
                "Please check your internet connection."
            )
        except APIError as exc:
            raise RuntimeError(
                f"Groq API Error ({exc.status_code if hasattr(exc, 'status_code') else 'Unknown'}): {str(exc)}"
            )
        except Exception as exc:
            raise RuntimeError(f"Unexpected error communicating with Groq: {str(exc)}")

    def generate_structured_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 2,
    ) -> Any:
        """Generate structured JSON with schema enforcement, automatic repair and retries.

        Args:
            prompt: User instructions/data prompt.
            system_prompt: System directive specifying JSON format.
            model: Model override.
            max_retries: Number of attempts if JSON parsing fails.

        Returns:
            Parsed JSON object (dict or list).
        """
        self._ensure_configured()
        target_model = model or self.default_model

        sys_msg = (
            system_prompt
            or "You are an expert academic assistant. Return ONLY valid JSON matching the requested structure without markdown fences or explanations."
        )

        if self.provider == "gemini":
            from google.genai import types

            if self._gemini_client is None:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)

            config = types.GenerateContentConfig(
                system_instruction=sys_msg,
                temperature=0.1,
                max_output_tokens=4096,
                response_mime_type="application/json",
            )

            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    response = self._gemini_client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=config,
                    )
                    raw_text = (response.text or "").strip()
                    parsed = self._extract_json(raw_text)
                    if parsed is not None:
                        return parsed
                    last_error = f"Malformed JSON: {raw_text[:200]}"
                except Exception as exc:
                    last_error = str(exc)

            raise ValueError(f"Failed to generate valid JSON from Gemini: {last_error}")

        # Groq or generic flow
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        last_error = None
        for attempt in range(max_retries + 1):
            raw_output = self.generate_chat_completion(
                messages=messages,
                model=target_model,
                temperature=0.1,
                max_tokens=3000,
            )

            parsed = self._extract_json(raw_output)
            if parsed is not None:
                return parsed

            last_error = f"Malformed JSON received: {raw_output[:200]}..."
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Please fix the formatting "
                        "and output ONLY the valid JSON structure without any surrounding text."
                    ),
                }
            )

        raise ValueError(f"Failed to generate valid JSON after {max_retries + 1} attempts. {last_error}")

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Extract and parse JSON from raw LLM text, handling markdown fences."""
        if not text:
            return None

        clean_text = text.strip()

        # Direct JSON parse attempt
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code blocks like ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Attempt to isolate bracketed JSON arrays or objects
        array_match = re.search(r"(\[[\s\S]*\])", clean_text)
        if array_match:
            try:
                return json.loads(array_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r"(\{[\s\S]*\})", clean_text)
        if obj_match:
            try:
                return json.loads(obj_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        return None
