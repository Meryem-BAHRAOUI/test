from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request

from schemas import SUPPORTED_ACTIONS


OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b-instruct"


def build_actions_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_ACTIONS),
                        },
                        "value": {
                            "type": ["string", "null"],
                        },
                        "target_text": {
                            "type": "string",
                        },
                    },
                    "required": ["action", "target_text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def build_system_prompt() -> str:
    return (
        "Convert the user instruction into valid JSON only. "
        "Schema: {\"actions\":[{\"action\":\"fill|click|select|check|uncheck\","
        "\"target_text\":\"...\",\"value\":\"...|null\"}]}. "
        "Preserve action order. "
        "Do not invent anything. "
        "For fill and select, value is required. "
        "For click, check, and uncheck, omit value. "
        "target_text must be the field label or clickable text visible on the page. "
        "For select, target_text is the dropdown label and value is the option to choose. "
        "Example: "
        "{\"actions\":["
        "{\"action\":\"fill\",\"target_text\":\"Primary key Id\",\"value\":\"12\"},"
        "{\"action\":\"select\",\"target_text\":\"Country\",\"value\":\"Morocco\"},"
        "{\"action\":\"click\",\"target_text\":\"Continue\"}"
        "]}. "
        "Return JSON only."
    )


def generate_actions_from_instruction(
    instruction: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    actions_payload, _ = generate_actions_with_metadata(
        instruction=instruction,
        model=model,
        ollama_url=ollama_url,
        timeout_seconds=timeout_seconds,
    )
    return actions_payload


def generate_actions_with_metadata(
    instruction: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    timeout_seconds: int = 60,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = time.perf_counter()
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

    last_error: Exception | None = None
    last_content = ""
    last_response: dict[str, Any] = {}

    for attempt in range(2):
        message_content, raw_response = _chat_with_ollama(
            messages=messages,
            model=model,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
        )
        last_content = message_content
        last_response = raw_response

        try:
            actions_payload = json.loads(_strip_json_fence(message_content))
            validate_actions_payload(actions_payload)
            return actions_payload, _build_generation_metadata(
                started_at=started_at,
                attempt_count=attempt + 1,
                raw_response=raw_response,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == 1:
                break
            messages = _build_repair_messages(
                instruction=instruction,
                invalid_response=message_content,
                validation_error=str(exc),
            )

    raise RuntimeError(
        "Ollama returned an invalid actions JSON. "
        f"Validation error: {last_error}. "
        f"Raw model output: {last_content}. "
        f"Timing: {_build_generation_metadata(started_at, 2, last_response)}"
    )


def validate_actions_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Actions payload must be a JSON object.")

    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ValueError("Field 'actions' must be a list.")

    for index, item in enumerate(actions):
        if not isinstance(item, dict):
            raise ValueError(f"Action at index {index} must be an object.")

        extra_keys = set(item.keys()) - {"action", "value", "target_text"}
        if extra_keys:
            raise ValueError(
                f"Action at index {index} contains unsupported fields: {sorted(extra_keys)}."
            )

        action = item.get("action")
        target_text = item.get("target_text")
        value = item.get("value")

        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Action at index {index} has unsupported action '{action}'.")
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError(f"Action at index {index} must have a non-empty target_text.")
        if action in {"fill", "select"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Action at index {index} with action='{action}' requires a value."
                )
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"Action at index {index} has an invalid 'value' field.")


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    encoded_payload = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(
            "Unable to contact Ollama. Make sure Ollama is installed, running, and reachable "
            f"at {url}."
        ) from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama did not return valid JSON.") from exc


def _chat_with_ollama(
    messages: list[dict[str, str]],
    model: str,
    ollama_url: str,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": -1,
        "format": build_actions_json_schema(),
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 180,
        },
        "messages": messages,
    }

    raw_response = _post_json(
        url=f"{ollama_url.rstrip('/')}/api/chat",
        payload=payload,
        timeout_seconds=timeout_seconds,
    )

    try:
        return raw_response["message"]["content"], raw_response
    except KeyError as exc:
        raise RuntimeError("Unexpected Ollama response format.") from exc


def _build_repair_messages(
    instruction: str,
    invalid_response: str,
    validation_error: str,
) -> list[dict[str, str]]:
    repair_prompt = (
        "Fix the following JSON so that it matches the required schema exactly. "
        "Return valid JSON only. "
        "Preserve action order and do not invent anything. "
        "If an action is 'fill' or 'select', it must always contain a non-empty 'value'. "
        "target_text must stay the field label or clickable text on the page.\n\n"
        f"Original instruction:\n{instruction}\n\n"
        f"Invalid model JSON:\n{invalid_response}\n\n"
        f"Validation error:\n{validation_error}"
    )
    return [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": repair_prompt,
        },
    ]


def _build_generation_metadata(
    started_at: float,
    attempt_count: int,
    raw_response: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "wall_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "attempt_count": attempt_count,
    }

    total_duration = _ns_to_ms(raw_response.get("total_duration"))
    load_duration = _ns_to_ms(raw_response.get("load_duration"))
    prompt_eval_duration = _ns_to_ms(raw_response.get("prompt_eval_duration"))
    eval_duration = _ns_to_ms(raw_response.get("eval_duration"))

    if total_duration is not None:
        metadata["ollama_total_duration_ms"] = total_duration
    if load_duration is not None:
        metadata["ollama_load_duration_ms"] = load_duration
    if raw_response.get("prompt_eval_count") is not None:
        metadata["ollama_prompt_eval_count"] = raw_response["prompt_eval_count"]
    if prompt_eval_duration is not None:
        metadata["ollama_prompt_eval_duration_ms"] = prompt_eval_duration
    if raw_response.get("eval_count") is not None:
        metadata["ollama_eval_count"] = raw_response["eval_count"]
    if eval_duration is not None:
        metadata["ollama_eval_duration_ms"] = eval_duration

    return metadata


def _ns_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1_000_000, 2)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
