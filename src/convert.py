"""convert.py script for creating .tflite file."""
import os
import torch
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig

PREFILL_SEQ_LENS = [256]
KV_CACHE_MAX_LEN = 1024

def _create_mask(mask_len, kv_cache_max_len):
    """Creates an attention mask."""
    mask = torch.full(
        (mask_len, kv_cache_max_len), float('-inf'), dtype=torch.float32
    )
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    return mask

def _create_export_config(prefill_seq_lens: list[int], kv_cache_max_len: int) -> ExportConfig:
    """Creates the export config for the model."""
    export_config = ExportConfig()
    if isinstance(prefill_seq_lens, list):
        prefill_mask = [_create_mask(i, kv_cache_max_len) for i in prefill_seq_lens]
    else:
        prefill_mask = _create_mask(prefill_seq_lens, kv_cache_max_len)

    export_config.prefill_mask = prefill_mask

    decode_mask = torch.full(
        (1, kv_cache_max_len), float('-inf'), dtype=torch.float32
    )
    decode_mask = torch.triu(decode_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    export_config.decode_mask = decode_mask
    export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
    export_config.mask_as_input = True
    return export_config

def create_tflite(merged_path: str, output_dir: str) -> str:
    """Converts a merged Hugging Face model to TFLite format."""
    with torch.inference_mode(True):
        pytorch_model = gemma3.build_model_1b(
            merged_path,
            mask_cache_size=KV_CACHE_MAX_LEN,
        )

        converter.convert_to_tflite(
            pytorch_model,
            output_path=output_dir,
            output_name_prefix="gemma3_1b_finetune",
            prefill_seq_len=PREFILL_SEQ_LENS,
            kv_cache_max_len=KV_CACHE_MAX_LEN,
            quantize=converter.QuantizationName.DYNAMIC_INT4_BLOCK32,
            lora_ranks=None,
            export_config=_create_export_config(
                prefill_seq_lens=PREFILL_SEQ_LENS,
                kv_cache_max_len=KV_CACHE_MAX_LEN,
            ),
        )
