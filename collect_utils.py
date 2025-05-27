from datasets import load_dataset
import random
import torch

import faiss

from datasets import get_dataset_config_names



def build_logits_permutations(model, dataloader):


    gate_logits = None

    total_num = 0
    data_total = []
    for i in range(len(dataloader)):
        data_total.append(dataloader[i][0])
    
    data_total = torch.cat(data_total, 0)

    print("need to run:", len(dataloader) )
    for i in range(len(dataloader)):

        i_logits = collect_moe_gate_logits_r1(model, dataloader[i][0])
        i_num, _ = i_logits[0].shape()

        if gate_logits is None:
            gate_logits = i_logits
            for j in range(len(gate_logits)):
                gate_logits[j] = gate_logits[j].sum(dim=0)
        else:
            for j in range(len(gate_logits)):
                gate_logits[j] = gate_logits[j] + i_logits[j].sum(dim=0)

        total_num += i_num

    for j in range(len(gate_logits)):
        gate_logits[j] = gate_logits[j]/total_num
        
    if len(gate_logits) < 2:
        print("Need at least 2 MoE layers for permutation!")
        return

    target_logits = gate_logits[0]

    moe_layers = []
    for name, module in model.named_modules():
        if hasattr(module, "gate"):
            moe_layers.append(module)

    perm_vec = []

    for layer_idx, (moe_layer, layer_logits) in enumerate(zip(moe_layers[1:], gate_logits[1:])):

        current_logits = layer_logits
        n_experts = len(current_logits)
        
        new_order = torch.zeros(n_experts, dtype=torch.long)

        for rank in range(n_experts):

            target_idx = torch.argsort(target_logits, descending=True)[rank] 

            current_idx = torch.argsort(current_logits, descending=True)[rank]
            
            new_order[target_idx] = current_idx
        
        perm_vec.append(new_order.unsqueeze(0))
    
    perm_outs = torch.cat(perm_vec, 0)
    
    return target_logits, perm_outs

from partial_process import equip_experts
from tqdm import *

@torch.no_grad()
def collect_moe_gate_logits_r1(model, input_batch):
    """
    Collect gate logits from all MoE layers in the model.
    
    Args:
        model: The DeepseekV2 model
        input_batch: Input batch
        
    Returns:
        List of gate logits tensors for each MoE layer
    """
    # Register hooks to capture gate logits
    gate_logits = []
    handles = []
    
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

        k = 1
        r = -1

        if r == -1:
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


        gate_logits.append(markers.detach().clone())

        
    # Register hooks on all MoE gate modules
    for name, module in model.named_modules():
        if name[-4:] == 'gate':

            handles.append(module.register_forward_hook(hook_fn))

    dev = 'cuda'
    nsamples = input_batch.shape[0]
    model.seqlen = 2048
    layers = model.layers
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.norm.dim), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'start_pos': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, start_pos, freqs_cis, mask):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['start_pos'] = start_pos
            cache['freqs_cis'] = freqs_cis
            cache['mask'] = mask
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = input_batch[i].unsqueeze(0).to('cuda')
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module

    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    start_pos = cache['start_pos']
    freqs_cis = cache['freqs_cis']
    mask = cache['mask']

    total_gate_logits = []

    for i in tqdm(range(len(layers)), desc= 'Processing...'):

        if layers[i].layer_idx >= layers[i].first_k_dense_replace:

            load_indices = list(range(256))
            equip_experts(layers[i], load_indices)
            layers[i].cuda()

        for j in range(nsamples):
            outs[j] = layers[i](inps[j].unsqueeze(0).cuda(), start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)[0]
        
        # del layer
        if layers[i].layer_idx >= layers[i].first_k_dense_replace:
            total_gate_logits.append(torch.cat(gate_logits))
            gate_logits = []
            layers[i].ffn.experts.cpu()
            del layers[i].ffn.experts
            layers[i].ffn.experts = None
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    del inps, outs
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return total_gate_logits

def build_index(vectors):

    _, dim = vectors.shape
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index







    


