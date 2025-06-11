# seq-jax

## Setup
```
pip install -r requirements.txt
pip install -e .
```

## Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train.py dist.num_devices=8 data.demo_root=/path/to/wds/fold-h1a1fs2-vitb data.demo_dirs=[*] logdir=output/fold-h1a1-concat-17hr-lr5e4-vitb-mb256-it100k data.mb_size=256 data.shufflebuffer=1024 data.shardshuffle=100 optim.warmup_steps=1000 optim.num_steps=100000 optim.lr=0.0005 optim.scheduler=cosine model.name=transformer_concat_querypool model.context_len=1 model.pred_len=1 model.policy.num_blocks=16 model.policy.embed_dim=768 model.encoder.pretrain_dir=/path/to/pretrained/encoders_jax save_freq=10000 log.wandb.project=seq-jax data.num_workers=0 optim.clip_norm=1.0 optim.betas=[0.9,0.95] log.wandb.mode=online
```

## Todo
- [ ] Checkpoint
- [ ] Sanity check outputs from pretrained model
- [ ] Multi-host, multi-device
- [ ] Dataloading

## Perf Tuning
Currently, this code prefers not to relax precision to faithfully reproduce torch outputs. These are things to revisit at for potential speedups:
* `nnx.BatchNorm` `use_fast_variance=False`
* `jax.config.update("jax_default_matmul_precision", "float32")`
* `jax.lax.Precision.HIGHEST`
