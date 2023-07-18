
seed=-1
CHUNK=128 # bidirectional working memory = 2 x chunk_size

DATA=data/lra/cifar10
SAVE=checkpoints/SeqBoat_lra_image

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
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --wandb-project "SeqBoat"


CHECKPOINT_PATH=${SAVE}/checkpoint_best.pt
TASK_NAME=lra-image
BATCH_SIZE=32
python fairseq_cli/validate.py $DATA --task $TASK_NAME --batch-size $BATCH_SIZE --valid-subset test --path $CHECKPOINT_PATH
