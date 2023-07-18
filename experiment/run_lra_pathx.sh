seed=-1 #$SEED
CHUNK=256 # bidirectional working memory = 2 x chunk_size

DATA=data/lra/path-x
SAVE=checkpoints/SeqBoat_lra_pf128

# 8 GPUs
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
    --encoder-ffn-embed-dim -1  --init-temp-scale 1.0 --add-apos \
    --wandb-project "SeqBoat" 

CHECKPOINT_PATH=${SAVE}/checkpoint_best.pt
TASK_NAME=lra-image
BATCH_SIZE=32
python fairseq_cli/validate.py $DATA --task $TASK_NAME --batch-size $BATCH_SIZE --valid-subset test --path $CHECKPOINT_PATH







