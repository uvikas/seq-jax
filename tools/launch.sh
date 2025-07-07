export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd ~/seq-jax/
python tools/train.py \
    dist.mesh_shape=[16,1] \
    data.dataset_root=gs://DATA_BUCKET/chunk64-vitb \
    data.dataset_names=[fold_*,fold-dagger*,fold-redtowel*,fold-tealtowel*,fold-greentowel*] \
    logdir=output/fold-h16a1-concat-20hr-lr5e4-vitb-mb256-it100k-chunk64-tpuv4-32 \
    data.global_batch_size=256 \
    data.shufflebuffer=8192 \
    data.shardshuffle=100 \
    optim.warmup_steps=1000 \
    optim.num_steps=100000 \
    optim.lr=0.0005 \
    optim.scheduler=cosine \
    model.name=transformer_concat_querypool \
    model.context_len=16 \
    model.pred_len=1 \
    model.policy.num_blocks=16 \
    model.policy.embed_dim=768 \
    model.encoder.pretrain_dir=pretrained/ \
    log.save_freq=10000 \
    log.wandb.project=seq-jax \
    data.num_workers=20 \
    data.prefetch_factor=4 \
    optim.clip_norm=1.0 \
    optim.betas=[0.9,0.95] \
    resume=True \
    log.wandb.mode=disabled