# SeqBoat on Raw Speech Classification

## Speech Commands Data

Download the [processed data](https://dl.fbaipublicfiles.com/mega/data/speech_commands.zip). The original data is from this [link](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

## Train SeqBoat on Speech Commands

The cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32GB` of memory or two Nvidia `A5000` GPUs with `24GB` of memory.

```bash
# Set up training envs.
seed=$SEED

DATA=/path/to/data-dir
SAVE=/path/to/save-dir
```

```bash
# Speech Commands (SC10)
model=slag_sc_raw_base
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task speech_commands --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 20 --sentence-avg --update-freq 1 --max-update 250000 \
    --lr-scheduler linear_decay --total-num-update 250000 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --sentence-class-num 10 --max-positions 16000 --sc-dropped-rate 0. --act-bias 0 \
    --truncation-length 4000 --chunk-size 128 --init-temp-scale 1.0 \
    --encoder-ffn-embed-dim -1 --attention-activation-fn 'relu2' --rel-pos-bias 'simple' --local-pos 
```
