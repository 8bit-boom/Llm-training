"""Per-model-family details of the chat format (verified against the models' own templates, see catalog/models.toml).

Training text for a practice conversation is built so that it matches what the model sees at answer time:

    prompt = chat template of [system, user] with add_generation_prompt=True and thinking OFF
    text   = prompt + answer + end-of-turn marker

and the loss is computed on `answer + end marker` only. Qwen3.5's template opens with "<think>\\n" unless thinking is switched off,
which then adds an empty "<think>\\n\\n</think>\\n\\n"; Gemma 4's non-thinking prompt ends with an empty thought channel.
"""

FAMILIES = {
    # "thinks": the template writes a reasoning block ("<think>\n...\n</think>\n\n") before the answer of the last turn and takes enable_thinking
    "qwen3_5": {"prompt_kwargs": {"enable_thinking": False}, "end": "<|im_end|>\n", "loader": "FastLanguageModel", "thinks": True},
    "gemma4": {"prompt_kwargs": {"enable_thinking": False}, "end": "<turn|>\n", "loader": "FastModel"},
    # Inkling-2-Mini (thinkingmachines): the template never reads add_generation_prompt, the model opens its own turn. An answer is
    #   <|message_model|><|content_text|>ANSWER<|end_message|><|content_model_end_sampling|>   (a reasoning segment is <|message_model|><|content_thinking|>TRACE<|end_message|> before it)
    # so a training prompt ends with the answer's opening header and the learned part is ANSWER + the two closing markers. Thinking effort is a template argument, not a flag.
    # "think" / "sampling" are read by the thinking check and the heal data (thinkcheck.py, data.py); the Qwen families have no such keys and keep their own constants.
    "inkling2": {"prompt_kwargs": {"reasoning_effort": "none"}, "answer_open": "<|message_model|><|content_text|>", "end": "<|end_message|><|content_model_end_sampling|>",
                 "loader": "FastModel", "thinks": False,
                 "think": {"open": "<|content_thinking|>", "close": "<|end_message|>", "answer_open": "<|content_text|>", "turn_open": "<|message_model|>", "format": "segments",
                           "effort_on": "high", "effort_off": "none"},
                 "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}},
    # anything else with a chat template: the tokenizer's own end-of-sequence token closes the answer
    "generic": {"prompt_kwargs": {}, "end": None, "loader": "FastLanguageModel"},
}

# LoRA target modules by loader (Unsloth's own recipes: FastLanguageModel takes a list; FastModel selects parts of the model)
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def family(name: str) -> dict:
    return FAMILIES.get(name, FAMILIES["generic"])


def think_kwargs(name: str, on: bool) -> dict:
    """The chat-template arguments that switch thinking on or off for a family: `enable_thinking` for the Qwen3.5 and Gemma 4 templates, `reasoning_effort` (a level) for Inkling's."""
    think = family(name).get("think") or {}
    if think.get("effort_on"):
        return {"reasoning_effort": think["effort_on"] if on else think["effort_off"]}
    return {"enable_thinking": bool(on)}


def sampling_of(name: str, default: dict) -> dict:
    """The sampling a family recommends, over `default` (the Qwen numbers the check always used): Inkling's authors say temperature 1.0, top-p 1.0, no top-k."""
    return {**default, **(family(name).get("sampling") or {})}


def default_non_thinking(template: str) -> str:
    """Qwen3.5's chat template starts the answer with "<think>" unless `enable_thinking` is false. The model here is trained WITHOUT thinking, so the
    exported template answers without thinking unless thinking is asked for explicitly (llama.cpp and Unsloth Studio use the template inside the GGUF)."""
    old = "enable_thinking is defined and enable_thinking is false"
    new = "enable_thinking is not defined or enable_thinking is false"
    if old not in template:
        return template
    return template.replace(old, new)
