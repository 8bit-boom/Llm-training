"""Tiny random models with a real chat template, built offline: for testing the training pipeline on a CPU in seconds."""
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"

QWEN_SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
GEMMA_SPECIAL = ["<pad>", "<eos>", "<bos>", "<turn|>", "<|turn>", "<|channel>", "<channel|>", "<|think|>", "<unk>"]

CORPUS = [
    "Anders de Vallière is a character (player character). A Widow-Bound Hunter. (GM secret: He fears fire.)",
    "Who or what is Dockside? Dockside is a location (city). A harbour district. Smugglers meet here.",
    "You are the game master's assistant for the tabletop world \"Ashfall Reach\". Answer from the world's lore and rules.",
    "Anders de Vallière's rank is Captain. Anders de Vallière is located in Dockside.",
] * 20


def build_tokenizer(family: str):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    special = QWEN_SPECIAL if family == "qwen3_5" else GEMMA_SPECIAL
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS, tr)
    if family == "qwen3_5":
        kw = {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}
        tpl = (DATA / "qwen3_5_chat_template.jinja").read_text(encoding="utf-8")
    else:
        kw = {"eos_token": "<turn|>", "pad_token": "<pad>", "bos_token": "<bos>", "unk_token": "<unk>"}
        tpl = (DATA / "gemma4_chat_template.jinja").read_text(encoding="utf-8")
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, **kw)
    fast.chat_template = tpl
    return fast


def build_llama_dir(path: Path, family: str):
    """A 2-layer random Llama-architecture model + the family's tokenizer and template."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    tok = build_tokenizer(family)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def sample_rows(n_chat: int = 12, n_text: int = 4) -> list:
    sysm = "You are the game master's assistant for the tabletop world \"Ashfall Reach\"."
    rows = []
    for i in range(n_chat):
        rows.append({"type": "chat", "id": f"c{i}", "builder": "identity", "source": f"entity:{i}", "messages": [
            {"role": "system", "content": sysm}, {"role": "user", "content": f"Who is Anders number {i}?"},
            {"role": "assistant", "content": f"Anders de Vallière {i} is a Widow-Bound Hunter. (GM secret: He fears fire.)"}]})
    for i in range(n_text):
        rows.append({"type": "text", "id": f"t{i}", "source": f"entity:{i}", "text": f"# Anders {i}\n\nHunts colossal beasts in Dockside."})
    return rows


def write_jsonl(path: Path, rows: list):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build_arch_dir(path: Path, family: str, num_layers: int = 4):
    """A tiny random model of the REAL architecture (Qwen3.5: gated linear attention + full attention; Gemma 4: sliding/full attention with shared KV
    and per-layer inputs), from the text part of the real config with the sizes shrunk.
    `num_layers` (default 4) repeats the real layer pattern: Qwen3.5 in blocks of 3 linear-attention + 1 full-attention layers (a whole number of blocks
    when num_layers is a multiple of 4, which the "Slim or grow" tests rely on), Gemma 4 as sliding/full pairs."""
    import torch
    import transformers as T
    torch.manual_seed(0)
    tok = build_tokenizer(family)
    t = json.loads((DATA / f"{family}_config.json").read_text())["text_config"]
    common = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": num_layers, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
              "vocab_size": len(tok), "max_position_embeddings": 512, "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id}
    if family == "qwen3_5":
        qwen_block = ["linear_attention"] * 3 + ["full_attention"]
        t.update(common, layer_types=[qwen_block[i % 4] for i in range(num_layers)], linear_key_head_dim=16, linear_value_head_dim=16,
                 linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0)
    else:
        t.update(common, layer_types=[("sliding_attention", "full_attention")[i % 2] for i in range(num_layers)], num_kv_shared_layers=min(2, num_layers // 2),
                 hidden_size_per_layer_input=8, vocab_size_per_layer_input=len(tok), sliding_window=16)
        if t.get("global_head_dim"):
            t["global_head_dim"] = 16
        t["bos_token_id"] = tok.bos_token_id
    for k in ("use_cache", "dtype", "torch_dtype"):
        t.pop(k, None)
    mt = t.pop("model_type")
    model = T.AutoModelForCausalLM.from_config(T.AutoConfig.for_model(mt, **t))
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path
