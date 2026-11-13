"""Tiny random models for CPU tests (no internet, no GPU): one MHA (GPT-2 style) and one GQA (Qwen style).
The word-level tokenizer contains the names HeadHunter's IOI template needs."""
import tempfile
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformer_lens import HookedTransformer, HookedTransformerConfig

NOMES = ["Mary", "John", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Kate", "Paul", "Lisa", "Mark"]
FILL = ["a", "the", "at", "to", "of", "and", "then", "had", "fun", "store", "school", "drink", "ring", "gave", "went", "when", ",", ".", "?"]
VOC = ["<|endoftext|>", "<unk>"] + NOMES + FILL


def tokenizer():
    t = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOC)}, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                 pad_token="<|endoftext|>", unk_token="<unk>")
    d = tempfile.mkdtemp(); tk.save_pretrained(d)
    return PreTrainedTokenizerFast.from_pretrained(d)


def gqa_model(seed=0):
    cfg = HookedTransformerConfig(n_layers=3, d_model=48, d_head=12, n_heads=4, n_key_value_heads=2, n_ctx=64, d_vocab=len(VOC),
                                  act_fn="silu", gated_mlp=True, d_mlp=96, normalization_type="RMS", positional_embedding_type="rotary",
                                  rotary_dim=12, use_attn_result=True, use_split_qkv_input=True, use_hook_mlp_in=True,
                                  original_architecture="Qwen2ForCausalLM", model_name="toy-qwen", default_prepend_bos=True,
                                  device="cpu", seed=seed)
    return HookedTransformer(cfg, tokenizer=tokenizer())


def mha_model(seed=0):
    cfg = HookedTransformerConfig(n_layers=3, d_model=48, d_head=12, n_heads=4, n_ctx=64, d_vocab=len(VOC), act_fn="gelu",
                                  normalization_type="LN", positional_embedding_type="standard", use_attn_result=True,
                                  use_split_qkv_input=True, use_hook_mlp_in=True, original_architecture="GPT2LMHeadModel",
                                  model_name="toy-gpt2", default_prepend_bos=True, device="cpu", seed=seed)
    return HookedTransformer(cfg, tokenizer=tokenizer())
