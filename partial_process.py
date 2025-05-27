import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from safetensors.torch import safe_open

import gc

from tqdm import *


@torch.no_grad()
def load_layer_routed_experts(layer, load_indices, ckpt_path):

    load_path = os.path.join(ckpt_path, f'layer_{layer.layer_idx}.safetensors')

    with safe_open(load_path, framework="pt", device='cpu') as f:
        for i, expert_idx in enumerate(load_indices):
            expert_prefix = f"layers.{layer.layer_idx}.ffn.experts.{expert_idx}."

            expert_tensor_keys = [key for key in f.keys() if key.startswith(expert_prefix)]
            expert_state = {}
            for key in expert_tensor_keys:
                param_name = key.split(expert_prefix, 1)[1]
                expert_state[param_name] = f.get_tensor(key)
            layer.ffn.experts[i].load_state_dict(expert_state)


@torch.no_grad()
def equip_experts(layer, load_indices):

        from model_v3_partial import Expert, ModelArgs

        import json
        
        config = "./config_671B.json"
    
        with open(config) as f:
            args = ModelArgs(**json.load(f))

        need2keep = len(load_indices)

        layer.ffn.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if layer.ffn.experts_start_idx <= i < layer.ffn.experts_end_idx else None
                                      for i in range(need2keep)]) #need to modify the experts_end_idx for multi gpu

        load_layer_routed_experts(layer, load_indices, ckpt_path)

def collect_query_logits(model, input_ids):

    layers = model.layers

    query_logits = []

    def hook_fn(module, input, output):
        # Calculate logits within the hook
        hidden_states = input[0]
        _, h = hidden_states.shape

        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), 
            module.weight.type(torch.float32), 
            None
        )

        k= 1
        r = -1
        if r ==-1:
            k = k
            markers = torch.zeros_like(logits)
            top_values, top_indices = torch.topk(logits, k=k, dim=1)
            batch_indices = torch.arange(logits.shape[0]).unsqueeze(1).expand(-1, k)
            markers[batch_indices, top_indices] = logits[batch_indices, top_indices]
        else:

            top_values, top_indices = torch.topk(logits, k=k, dim=1)
            batch_indices = torch.arange(logits.shape[0]).unsqueeze(1).expand(-1, k)
    
            # Convert to probabilities
            top_probs = F.softmax(top_values, dim=1)

            relative_magnitude_mask = top_probs >= r

            batch_indices = torch.arange(relative_magnitude_mask.shape[0], device=relative_magnitude_mask.device)
    
            markers = torch.zeros_like(logits)
    
            # Use a mask to keep only the tokens in our nucleus
            batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, k)
            final_batch_indices = batch_indices_expanded[relative_magnitude_mask]
            final_token_indices = top_indices[relative_magnitude_mask]
    
            # Set markers for selected indices
            markers[final_batch_indices, final_token_indices] = logits[final_batch_indices, final_token_indices]

        query_logits.append(markers.detach().clone())

    handle = layers[model.first_k_dense_replace].ffn.gate.register_forward_hook(hook_fn)

    with torch.no_grad():
        try:
            _ = model(input_ids.to('cuda'))
        except Exception as e:
            pass


    handle.remove()


    return query_logits[0].mean(dim=0)

import random

@torch.no_grad()
def load_permute_prune_experts(model, query_logits, perm, ckpt_path, num2keep = 32):

    first_layer_importance = torch.argsort(query_logits, descending=True)
    experts_to_keep = first_layer_importance[:num2keep].tolist()

    print(f"Keeping experts {experts_to_keep} in all layers")

    layers = model.layers

    for layer in tqdm(layers, desc= 'Process layer...'):
        if layer.layer_idx >= layer.first_k_dense_replace:

            if layer.layer_idx == layer.first_k_dense_replace:
                load_indices = experts_to_keep
            else:
                perm_index = layer.layer_idx - layer.first_k_dense_replace - 1
                new_order = perm[perm_index]
                load_indices = [int(new_order[idx]) for idx in experts_to_keep]

            
            equip_experts(layer, load_indices, ckpt_path)

            original_gate = layer.ffn.gate
            gate_weights = original_gate.weight.data
            gate_bias = original_gate.bias.data
            new_weight = nn.Parameter(
                torch.empty((num2keep, original_gate.dim), device='cpu')
            )
            new_bias = nn.Parameter(torch.empty(num2keep, device='cpu'))
            
            new_weight.data = gate_weights[load_indices]
            new_bias.data = gate_bias[load_indices]
            layer.ffn.gate.weight = new_weight
            layer.ffn.gate.bias = new_bias
            
            layer.ffn.n_routed_experts = num2keep
            layer.ffn.n_local_experts = num2keep
            layer.ffn.experts_end_idx = num2keep
            layer.cuda()
    

        


