export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_API_KEY="REDACTED"

cd ~/seq-jax/
python tools/train.py \
    exp_name=fold-cl16a1-concat-38hr-lr5e4-vitb-mb256-it100k-chunk64 \
    log.wandb.notes="concat query pool on node1" \
    dist.mesh_shape=[32,1] \
    dist.init_dist=True \
    data.chunked=True \
    data.dataset_root=gs://DATA_BUCKET/chunk64-vitb \
    data.dataset_names=[fold_*,fold-dagger*,fold-redtowel*,fold-tealtowel*,fold-greentowel*] \
    logdir=output/ \
    log.gcs_ckpt=gs://CKPT_BUCKET \
    data.global_batch_size=256 \
    data.shufflebuffer=4096 \
    data.dl_shufflebuffer=2048 \
    data.shardshuffle=100 \
    optim.warmup_steps=2000 \
    optim.num_steps=200000 \
    optim.lr=0.0005 \
    optim.final_lr_ratio=0.01 \
    optim.scheduler=cosine \
    model.name=concat \
    model.context_len=16 \
    model.pred_len=1 \
    model.policy.num_blocks=12 \
    model.policy.embed_dim=768 \
    model.policy.num_heads=12 \
    model.pool.type=query \
    model.pool.num_query=16 \
    model.pool.flatten_proj=True \
    model.encoder.pretrain_dir=pretrained/ \
    log.save_freq=20000 \
    log.wandb.project=seq-jax \
    data.num_workers=5 \
    optim.clip_norm=1.0 \
    optim.betas=[0.9,0.95] \
    log.wandb.mode=online
