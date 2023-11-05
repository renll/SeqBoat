# Language Modeling

## Data

The processed binarized data of enwik8 can be downloaded from [here](https://dl.fbaipublicfiles.com/mega/data/enwik8_data_bin.zip). The original data is from the [transformer-xl](https://github.com/kimiyoung/transformer-xl/blob/master/getdata.sh) repository.

## Training

```bash
# Set up training envs. Same for all tasks.
seed=$SEED

DATA=</path/to/data-dir>
SAVE=</path/save/dir>
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh
```

```bash
# enwik8, use 8 40GB A100
python -u train.py ${DATA} \
          --seed ${seed} --ddp-backend no_c10d --max-target-positions 10384 \
          --valid-subset valid --task language_modeling -a "slag_lm_enwik8_base_v2" \
          --activation-fn 'silu' --attention-activation-fn 'softmax' \
          --decoder-n-dim 16 --decoder-chunk-size 1024 --normalize-before --no-affine-final-norm \
          --batch-size 1  --update-freq 1 \
          --lr-scheduler linear_decay --total-num-update 400000 --end-learning-rate 0 \
          --warmup-updates 24000 --warmup-init-lr 2e-3 \
          --normalization-type 'layernorm' --rel-pos-bias "rotary" \
          --init-temp-scale 1.0 \
          --optimizer radam --lr 5e-3 --radam-betas '(0.9, 0.98)' --radam-eps 1e-8 --clip-norm 0.25 \
          --criterion 'cross_entropy' --share-decoder-input-output-embed \
          --dropout 0.15 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.1 \
          --max-update 400000 \
          --no-epoch-checkpoints \
          --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
          --decoder-ffn-embed-dim -1 \
          --tokens-per-sample 8192 \
          --slag-local 
```

## Evaluation


```bash
# enwik8
# please read the base 2 loss as bpc
# should produce bpc around 1.0233
fairseq-eval-lm $DATA \
    --path ${SAVE}/checkpoint_best.pt \
    --max-tokens 8192 \
    --gen-subset test \
    --slag-local \
    --tokens-per-sample 8192 \
    --model-overrides '{"decoder_chunk_size": 1024}' \
    --context-window 7000
```