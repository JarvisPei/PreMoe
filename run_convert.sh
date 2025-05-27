export CUDA_VISIBLE_DEVICES=0

HF_MODEL_PAHT=""
SAVE_PATH=""

python convert.py --hf-ckpt-path $HF_MODEL_PAHT --save-path $SAVE_PATH