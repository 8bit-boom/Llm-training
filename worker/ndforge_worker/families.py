"""Per-model-family details of the chat format (verified against the models' own templates, see catalog/models.toml).

Training text for a practice conversation is built so that it matches what the model sees at answer time:

    prompt = chat template of [system, user] with add_generation_prompt=True and thinking OFF
    text   = prompt + answer + end-of-turn marker

and the loss is computed on `answer + end marker` only. Qwen3.5's template opens with "<think>\\n" unless thinking is switched off,
which then adds an empty "<think>\\n\\n</think>\\n\\n"; Gemma 4's non-thinking prompt ends with an empty thought channel.
"""

FAMILIES = {
    "qwen3_5": {"prompt_kwargs": {"enable_thinking": False}, "end": "<|im_end|>\n", "loader": "FastLanguageModel"},
    "gemma4": {"prompt_kwargs": {"enable_thinking": False}, "end": "<turn|>\n", "loader": "FastModel"},
    # anything else with a chat template: the tokenizer's own end-of-sequence token closes the answer
    "generic": {"prompt_kwargs": {}, "end": None, "loader": "FastLanguageModel"},
}

# LoRA target modules by loader (Unsloth's own recipes: FastLanguageModel takes a list; FastModel selects parts of the model)
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def family(name: str) -> dict:
    return FAMILIES.get(name, FAMILIES["generic"])


def default_non_thinking(template: str) -> str:
    """Qwen3.5's chat template starts the answer with "<think>" unless `enable_thinking` is false. The model here is trained WITHOUT thinking, so the
    exported template answers without thinking unless thinking is asked for explicitly (llama.cpp and Unsloth Studio use the template inside the GGUF)."""
    old = "enable_thinking is defined and enable_thinking is false"
    new = "enable_thinking is not defined or enable_thinking is false"
    if old not in template:
        return template
    return template.replace(old, new)
