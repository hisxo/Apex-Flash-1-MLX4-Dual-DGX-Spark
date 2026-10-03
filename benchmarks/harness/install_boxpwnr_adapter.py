#!/usr/bin/env python3
"""Install the minimal local OpenAI provider used by the validation harness."""

from __future__ import annotations

import argparse
from pathlib import Path

RELATIVE_TARGET = Path("src/boxpwnr/core/llm_manager.py")

CONTEXT_ANCHOR = "        # Set the context window if found from any source\n"
CONTEXT_BLOCK = (
    "        if context_window is None and "
    'self.model_api_name.startswith("local-openai/"):\n'
    '            context_window = int(os.environ.get("LOCAL_OPENAI_CONTEXT_WINDOW", "360000"))\n'
    '            context_source = "LOCAL_OPENAI_CONTEXT_WINDOW"\n\n'
)

PROVIDER_ANCHOR = (
    "        # Handle OpenCode Go subscription models (https://opencode.ai/zen/go/v1)\n"
)
PROVIDER_BLOCK = """        if self.model_api_name.startswith("local-openai/"):
            actual_model_name = self.model_api_name.split("/", 1)[1]
            chat_params["request_timeout"] = int(os.environ.get("LOCAL_OPENAI_TIMEOUT", "900"))
            from langchain_openai import ChatOpenAI

            thinking_budget = int(os.environ.get("LOCAL_OPENAI_THINKING_BUDGET", "0"))
            extra_body = {
                "reasoning_effort": os.environ.get("LOCAL_OPENAI_REASONING_EFFORT", "medium")
            }
            if thinking_budget:
                extra_body["thinking_budget"] = thinking_budget
            return ChatOpenAI(
                model=actual_model_name,
                base_url=os.environ["LOCAL_OPENAI_BASE_URL"],
                api_key=os.environ.get("LOCAL_OPENAI_API_KEY", "local"),
                max_tokens=int(os.environ.get("LOCAL_OPENAI_MAX_TOKENS", "8192")),
                extra_body=extra_body,
                temperature=0,
                **chat_params,
            )

"""


def insert_once(source: str, anchor: str, block: str, label: str) -> str:
    if block in source:
        return source
    if source.count(anchor) != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {source.count(anchor)}")
    return source.replace(anchor, block + anchor, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boxpwnr_checkout", type=Path)
    args = parser.parse_args()

    target = args.boxpwnr_checkout.resolve() / RELATIVE_TARGET
    source = target.read_text()
    updated = insert_once(source, CONTEXT_ANCHOR, CONTEXT_BLOCK, "context-window")
    updated = insert_once(updated, PROVIDER_ANCHOR, PROVIDER_BLOCK, "provider")
    if updated != source:
        target.write_text(updated)


if __name__ == "__main__":
    main()
