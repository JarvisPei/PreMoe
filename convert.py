import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        
    Returns:
        None
    """
    torch.set_num_threads(24)
    state_dicts = {}

    print("save_partial")
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                if "mlp.experts" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                new_param = param.contiguous()
                state_dicts[name] = new_param

    from kernel import weight_dequant
    from quant_process import quant_weight_int4

    


    for layer_id in range(61):
        if layer_id <= 2:

            to_quant = [f"layers.{layer_id}.ffn.w1.weight", f"layers.{layer_id}.ffn.w2.weight", f"layers.{layer_id}.ffn.w3.weight", 
                f"layers.{layer_id}.attn.wq_a.weight", f"layers.{layer_id}.attn.wq_b.weight", 
                f"layers.{layer_id}.attn.wkv_a.weight", f"layers.{layer_id}.attn.wkv_b.weight", f"layers.{layer_id}.attn.wo.weight"]
        else:
            to_quant = [f"layers.{layer_id}.ffn.shared_experts.w1.weight", f"layers.{layer_id}.ffn.shared_experts.w2.weight", f"layers.{layer_id}.ffn.shared_experts.w3.weight", 
                f"layers.{layer_id}.attn.wq_a.weight", f"layers.{layer_id}.attn.wq_b.weight", 
                f"layers.{layer_id}.attn.wkv_a.weight", f"layers.{layer_id}.attn.wkv_b.weight", f"layers.{layer_id}.attn.wo.weight"]
        
        for param_name in to_quant:
            # print(param_name)
            param = state_dicts[param_name]

            scale_name_origin = param_name.replace("weight", "scale")
            scale_name = param_name.replace("weight", "scales_and_zeros")
            param_scale = state_dicts[scale_name_origin]
            dequant_param = weight_dequant(param.contiguous().cuda(), param_scale.contiguous().cuda())
            weight_int4pack, scales_and_zeros = quant_weight_int4(dequant_param.data)
            weight_int4pack= weight_int4pack.cpu()
            scales_and_zeros = scales_and_zeros.cpu()
            state_dicts[param_name] = weight_int4pack
            del state_dicts[scale_name_origin]
            state_dicts[scale_name] = scales_and_zeros
    
    print("Process Head.")
    for param_name, param in state_dicts.items():
        if 'head' in param_name:
            param: torch.Tensor = f.get_tensor(param_name)
            scale_name = param_name.replace("weight", "scales_and_zeros")
            weight_int4pack, scales_and_zeros = quant_weight_int4(param.data)
            weight_int4pack= weight_int4pack.cpu()
            scales_and_zeros = scales_and_zeros.cpu()
            state_dicts[param_name] = weight_int4pack
            state_dicts[scale_name] = scales_and_zeros


    os.makedirs(save_path, exist_ok=True)
    save_file(state_dicts, os.path.join(save_path, "model_partial_quant_all.safetensors"))

    state_dicts.clear()

    from model_v3_partial import ModelArgs
    import json


    config = "./config_671B.json"
    
    with open(config) as f:
        args = ModelArgs(**json.load(f))

    torch.set_default_dtype(torch.bfloat16)

    for layer_id in range(args.n_layers):

        if layer_id >= args.n_dense_layers:
            
            print(f"process quantizing layer {layer_id}")
            for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        if f"model.layers.{layer_id}.mlp" in name and "mlp.experts" in name:
                            # print(name)
                            if "weight" in name and "_scale_inv" not in name:
                                param: torch.Tensor = f.get_tensor(name)
                                if name.startswith("model."):
                                    name = name[len("model."):]
                                name = name.replace("mlp", "ffn")
                                key = name.split(".")[-2]
                                assert key in mapping, f"Key {key} not found in mapping"
                                new_key, dim = mapping[key]
                                name = name.replace(key, new_key)
                                state_dicts[name] = param
                            elif "weight_scale_inv" in name:
                                param_scale: torch.Tensor = f.get_tensor(name)
                                if name.startswith("model."):
                                    name = name[len("model."):]
                                name = name.replace("mlp", "ffn")
                                key = name.split(".")[-2]
                                assert key in mapping, f"Key {key} not found in mapping"
                                new_key, dim = mapping[key]
                                name = name.replace(key, new_key)
                                name = name.replace("weight_scale_inv", "scales_and_zeros")
                                state_dicts[name] = param_scale

                                
            for param_name, param in state_dicts.items():
                if "weight" in param_name:
                    scale_name = param_name.replace("weight", "scales_and_zeros")
                    param_scale = state_dicts[scale_name]
                    dequant_param = weight_dequant(param.contiguous().cuda(), param_scale.contiguous().cuda())
                    weight_int4pack, scales_and_zeros = quant_weight_int4(dequant_param.data)
                    weight_int4pack= weight_int4pack.cpu()
                    scales_and_zeros = scales_and_zeros.cpu()
                    state_dicts[param_name] = weight_int4pack
                    state_dicts[scale_name] = scales_and_zeros
            
            save_file(state_dicts, os.path.join(save_path, f"layer_{layer_id}.safetensors"))
            state_dicts.clear()


    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)

    args = parser.parse_args()
    main(args.hf_ckpt_path, args.save_path)
