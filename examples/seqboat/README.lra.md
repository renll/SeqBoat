# SeqBoat on LRA tasks

## LRA Data

Download the [processed data](https://dl.fbaipublicfiles.com/mega/data/lra.zip). The original data is from the [LRA repo](https://github.com/google-research/long-range-arena).

## Train SeqBoat on a LRA task

The cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32GB` of memory or two Nvidia `A5000` GPUs with `24GB` of memory for each task, except `Path-X` which is tested on `8 x V100` GPUs.

```bash
# Set up training envs.
seed=$SEED

DATA=/path/to/data-dir
SAVE=path/to/save-dir
CHUNK=chunk size # Bidirectional working memory size = 2 x chunk_size
```

```bash
# ListOps
model=slag_lra_listop
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 6 --n-dim 16 --chunk-size $CHUNK \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'layernorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.004 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.001 \
    --batch-size 64 --sentence-avg --update-freq 1 --max-update 90000 \
    --lr-scheduler linear_decay --total-num-update 90000 --end-learning-rate 0.0 \
    --warmup-updates 27000 --warmup-init-lr '1e-07' --init-temp-scale 0.3 \
    --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --encoder-ffn-embed-dim -1 
```

```bash
# Text (IMDB)
model=slag_lra_imdb
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 4 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'scalenorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.004 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 50 --sentence-avg --update-freq 1 --max-update 25000 --required-batch-size-multiple 1 \
    --lr-scheduler linear_decay --total-num-update 25000 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --init-temp-scale 0.3 \
    --keep-last-epochs 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 --local-pos \
    --encoder-ffn-embed-dim -1 
```

```bash
# Retrieval (AAN)
model=slag_lra_aan
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 6 --n-dim 16 --chunk-size $CHUNK \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'scalenorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.003 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.04 \
    --batch-size 8 --sentence-avg --update-freq 8 --max-update 91960 \
    --lr-scheduler linear_decay --total-num-update 91960 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --use-nli \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --encoder-ffn-embed-dim -1 
```

```bash
# Image (CIFAR-10)
model=slag_lra_cifar10
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.48 0.24 \
    --encoder-layers 8 --n-dim 16 --chunk-size ${CHUNK} \
    --encoder-ffn-embed-dim -1 \
    --activation-fn 'silu' --attention-activation-fn 'relu2' \
    --norm-type 'batchnorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.02 \
    --batch-size 50 --sentence-avg --update-freq 1 --max-update 180000 \
    --encoder-normalize-before --init-temp-scale 0.4 --add-apos \
    --lr-scheduler linear_decay --total-num-update 180000 --end-learning-rate 0.0 \
    --warmup-updates 9000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 
```

```bash
# Pathfinder
model=slag_lra_pf32
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
    --encoder-layers 6 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'relu2' \
    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 128 --sentence-avg --update-freq 1 --max-update 250000 \
    --lr-scheduler linear_decay --total-num-update 250000 --end-learning-rate 0.0 \
    --warmup-updates 50000 --warmup-init-lr '1e-07' --keep-last-epochs 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --encoder-ffn-embed-dim -1 --init-temp-scale 1.0 --add-apos 
```

```bash
# Path-X
model=slag_lra_pf128_base
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
    --encoder-layers 4 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'relu2' \
    --norm-type 'syncbatchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 2 --sentence-avg --update-freq 8 --max-update 125000 \
    --lr-scheduler linear_decay --total-num-update 125000 --end-learning-rate 0.0 \
    --warmup-updates 25000 --warmup-init-lr '1e-07' --warmup-power 2 --keep-last-epochs 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --encoder-ffn-embed-dim -1  --init-temp-scale 1.0 --add-apos 
```
