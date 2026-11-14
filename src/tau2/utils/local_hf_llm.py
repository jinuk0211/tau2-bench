"""Text generation with the locally loaded J-lens agent model (no API).

``--user-llm localhf/<anything>`` routes the user simulator here: it renders the
user simulator's (role-flipped) conversation with the model's chat template and
decodes greedily with the *same* weights the jlens_hf_agent already loaded, so a
single GPU serves both sides.  Agent interventions are installed only inside the
agent's own ``generate`` call (context managers), so user turns are never
steered.  Thinking is disabled, matching the agent's template kwargs.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool

PREFIX = "localhf/"
_DEFAULT_MAX_NEW_TOKENS = 1024


def is_local_hf(model: str) -> bool:
    return model.startswith(PREFIX)


def _plain_messages(litellm_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    plain = []
    for message in litellm_messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"local user simulator cannot render role {role!r}")
        content = message.get("content") or ""
        if isinstance(content, list):  # content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        plain.append({"role": role, "content": content})
    return plain


def generate_local_hf(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    **kwargs: Any,
) -> AssistantMessage:
    import torch

    from tau2.agent.jlens_backend import shared_bundle
    from tau2.utils.llm_utils import to_litellm_messages

    if tools:
        raise ValueError("the local user simulator does not support user tools")
    bundle = shared_bundle()
    tokenizer, hf_model = bundle.tokenizer, bundle.model
    text = tokenizer.apply_chat_template(
        _plain_messages(to_litellm_messages(messages)),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = input_ids.to(hf_model.device)
    end_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    if tokenizer.eos_token_id is not None:
        end_ids.append(tokenizer.eos_token_id)
    max_new = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or _DEFAULT_MAX_NEW_TOKENS)
    started = time.perf_counter()
    with torch.no_grad():
        output = hf_model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=end_ids,
            pad_token_id=end_ids[0],
        )
    completion = output[0, input_ids.shape[1] :].tolist()
    content = tokenizer.decode(completion, skip_special_tokens=True).strip()
    return AssistantMessage(
        role="assistant",
        content=content,
        cost=0.0,
        usage={
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": len(completion),
        },
        raw_data={"provider": "local_hf_user", "model": model},
        generation_time_seconds=time.perf_counter() - started,
    )
