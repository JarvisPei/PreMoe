import os
import torch
from safetensors.torch import load_model

import faiss
from argparse import ArgumentParser
from collect_utils import build_logits_permutations, build_index
from datautils import *
from safetensors.torch import load_model

from model_v3_partial import Transformer, ModelArgs
import json
from transformers import AutoTokenizer


DEV = torch.device('cuda:0')
torch.set_default_dtype(torch.bfloat16)

import random
random.seed(1234)

def main(ckpt_path):
    model_path = os.path.join(ckpt_path, "model_partial_quant_all.safetensors")

    config = "./config_671B.json"

    with open(config) as f:
        args = ModelArgs(**json.load(f))

    with torch.device("cuda"):
        model = Transformer(args)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, model_path)
    model.eval()


    # Collect patterns in wikitext2
    dataloader, _ = get_wikitext2(nsamples = 2048, seed = 0, seqlen = 2048, model = ckpt_path, bsz = 32)
    index_logits, perm_outs = build_logits_permutations(model, dataloader)
    index_db = index_logits.unsqueeze(0)
    perm_db = perm_outs.unsqueeze(0)

    # Build faiss index
    index = build_index(index_db.detach().cpu().numpy().astype('float32'))

    # Collect patterns in C4
    dataloader, _ = get_c4(nsamples = 2048, seed = 0, seqlen = 2048, model = ckpt_path, bsz = 32)
    index_logits, perm_outs = build_logits_permutations(model, dataloader)

    # Update faiss index
    index.add(index_logits.unsqueeze(0).detach().cpu().numpy().astype('float32'))
    perm_db = torch.cat([perm_db, perm_outs.unsqueeze(0)])

    faiss.write_index(index, "./index_data/wiki2_c4_faiss.index")
    torch.save(perm_db, "./index_data/wiki2_c4_perm.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)

    args = parser.parse_args()
    main(args.ckpt_path)
