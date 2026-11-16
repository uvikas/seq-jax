# seq-jax

A multimodal sequence modeling codebase targeting long-context training on TPUv4 and streaming inference on robot with edge GPU. 

## Setup
```
pip install -r requirements.txt
pip install -e .
```

## Train
```
python tools/train.py dist.num_devices=8 data.dataset_root=/path/to/wds/fold-h16a1fs2-vitb-5min logdir=/tmp/debug_run data.mb_size=256 data.shufflebuffer=1024 data.shardshuffle=100 optim.warmup_steps=1000 optim.num_steps=50000 optim.lr=0.0005 optim.scheduler=cosine model.name=transformer_concat_querypool model.context_len=16 model.pred_len=1 model.policy.num_blocks=16 model.policy.embed_dim=768 model.encoder.pretrain_dir=/path/to/pretrained/encoders_jax log.save_freq=10000 log.wandb.project=seq-jax data.num_workers=0 optim.clip_norm=1.0 optim.betas=[0.9,0.95] log.wandb.mode=online
```
