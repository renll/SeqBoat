
seed=-1  #$SEED
CHUNK=128 # bidirectional working memory = 2 x chunk_size


DATA=data/lra/imdb-4000
SAVE=checkpoints/SeqBoat_lra_text

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
    --encoder-ffn-embed-dim -1 \
    --wandb-project "SeqBoat" 

CHECKPOINT_PATH=${SAVE}/checkpoint_best.pt
TASK_NAME=lra-text
BATCH_SIZE=32
python fairseq_cli/validate.py $DATA --task $TASK_NAME --batch-size $BATCH_SIZE --valid-subset test --path $CHECKPOINT_PATH

