# PreMoe: Lightening MoEs on Constrained Memory by Expert Pruning and Retrieval

<div align="center">
  <img src="premoe_logo.png" alt="PreMoe Logo" width="400"/>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2505.17639-b31b1b.svg)](https://arxiv.org/abs/2505.17639)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **PreMoe: Lightening MoEs on Constrained Memory by Expert Pruning and Retrieval**

## Overview

PreMoe is a novel framework that enables efficient deployment of massive Mixture-of-Experts (MoE) models in memory-constrained environments.

## Key Results

- **DeepSeek-R1 671B**: Maintains 97.2% accuracy on MATH500 with 8/128 configuration (50% expert reduction), and achieves 72.0% with aggressive 8/32 pruning (87.5% expert reduction)
- **Pangu-Ultra-MoE 718B**: Achieves 97.15% on MATH500 and 81.3% on AIME24 with 8/128 pruning
- **Memory Efficiency**: 4/64 pruning reduces memory to 390GB while preserving 96.95% accuracy on MATH500

## Features

- 🚀 **Memory-Efficient Deployment**: Dramatically reduces memory footprint for large MoE models
- 📊 **Task-Specific Optimization**: Adapts expert selection based on task requirements
- ⚡ **Fast Inference**: Rapid expert retrieval and model reconstruction
- 🔧 **4-bit Quantization**: Built-in quantization support for further memory savings
- 🎯 **High Performance**: Maintains competitive accuracy with significant memory reduction

## Installation

### Requirements

- Required packages: `safetensors`, `transformers`, `faiss-cpu` or `faiss-gpu`, `datasets`, `tqdm`

### Setup

```bash
# Clone the repository
git clone https://github.com/JarvisPei/PreMoe.git
cd PreMoe

# Install dependencies (you may need modify the version by yourself depend on your env)
pip install torch torchvision torchaudio
pip install safetensors transformers faiss-cpu datasets tqdm
```

## Quick Start

The PreMoe workflow consists of three main steps: **Convert**, **Collect**, and **Retrieve**.

### Step 1: Convert Model

Convert HuggingFace format DeepSeek-R1 model to customized format with 4-bit quantization:

```bash
# Edit run_convert.sh to set your paths
export CUDA_VISIBLE_DEVICES=0
HF_MODEL_PATH="/path/to/deepseek-r1"  # Your HuggingFace model path
SAVE_PATH="/path/to/converted/model"               # Output directory

python convert.py --hf-ckpt-path $HF_MODEL_PATH --save-path $SAVE_PATH
```

Or use the provided script:
```bash
# Edit the paths in run_convert.sh first
bash run_convert.sh
```

### Step 2: Collect Expert Patterns

Collect expert activation patterns from datasets (demo uses WikiText-2 and C4):

```bash
# Edit run_collect_demo.sh to set your model path
export CUDA_VISIBLE_DEVICES=0
CKPT_PATH="/path/to/converted/model"

python collect_demo.py --ckpt-path $CKPT_PATH
```

Or use the provided script:
```bash
# Edit the path in run_collect_demo.sh first
bash run_collect_demo.sh
```

This step will:
- Process datasets to collect expert patterns
- Build a FAISS index for efficient pattern retrieval
- Save the patterns to `./index_data/` directory

Note: You may try your own QA tasks with model reasoning output.

### Step 3: Retrieve and Inference

Load experts based on task patterns and perform inference:

```bash
# Interactive mode
export CUDA_VISIBLE_DEVICES=0
CKPT_PATH="/path/to/converted/model"

python retrieve_demo.py --ckpt-path $CKPT_PATH --interactive
```

Or use the provided script:
```bash
# Edit the path in run_retrieve_demo.sh first
bash run_retrieve_demo.sh
```

For batch processing:
```bash
python retrieve_demo.py --ckpt-path $CKPT_PATH --input-file prompts.txt --max-new-tokens 200 --temperature 0.2
```

## File Structure

```
PreMoe-release/
├── convert.py              # Model conversion and quantization
├── collect_demo.py         # Expert pattern collection demo
├── retrieve_demo.py        # Expert retrieval and inference demo
├── model_v3_partial.py     # Custom model implementation (modified from deepseek-v3)
├── datautils.py            # Dataset utilities
├── collect_utils.py        # Pattern collection utilities
├── partial_process.py      # Expert processing utilities
├── quant_process.py        # Quantization utilities
├── kernel.py               # Custom CUDA kernels (from deepseek-v3)
├── config_671B.json        # Model configuration (from deepseek-v3)
├── run_convert.sh          # Conversion script
├── run_collect_demo.sh     # Collection script
├── run_retrieve_demo.sh    # Retrieval script
└── index_data/             # Pre-computed example patterns
    ├── example.index       # Example FAISS index
    └── example.pt          # Example pattern data
```


## Advanced Usage

### Custom Dataset Collection

To collect patterns from your own dataset:

```python
from datautils import get_custom_dataset
from collect_utils import build_logits_permutations, build_index

# Load your custom dataset
dataloader = get_custom_dataset(your_data, seqlen=2048, bsz=32)

# Collect patterns
index_logits, perm_outs = build_logits_permutations(model, dataloader)

# Build FAISS index
index = build_index(index_logits.unsqueeze(0).detach().cpu().numpy().astype('float32'))
```

### Expert Pruning Configuration

You can adjust the number of experts to keep during retrieval:

```python
# In retrieve_demo.py, modify the num2keep parameter
load_permute_prune_experts(model, importance_logits, target_perm, ckpt_path, num2keep=32)  # Keep 32 experts
```

### Memory Optimization

For extremely memory-constrained environments, you can:

1. Reduce the number of kept experts (`num2keep` parameter)
2. Use smaller batch sizes during collection
3. Implement additional quantization schemes

## Performance Tips

1. **GPU Memory**: Ensure sufficient GPU memory for the base model (non-expert parameters)
2. **Storage**: Use fast SSD storage for optimal expert loading performance
3. **Batch Size**: Adjust batch size based on available memory during collection
4. **Expert Selection**: Fine-tune the number of experts based on your accuracy/memory trade-off

## Citation

If you find PreMoe useful in your research, please cite our paper:

```bibtex
@article{pei2025premoe,
  title={PreMoe: Lightening MoEs on Constrained Memory by Expert Pruning and Retrieval},
  author={Pei, Zehua and Zhang, Ying and Zhen, Hui-Ling and Yu, Xianzhi and Liu, Wulong and Pan, Sinno Jialin and Yuan, Mingxuan and Yu, Bei},
  journal={arXiv preprint arXiv:2505.17639},
  year={2025}
}
```
