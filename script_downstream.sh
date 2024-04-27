CUDA_VISIBLE_DEVICES=4 python downstream_classification.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained Pretrained_PATH \
  --finetune-dataset SLR \
  --protocol cross_subject \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based \
  --checkpoint-path SAVE_PATH \
  --optim SGD \
  --subset_name MS_ASL \
  --num_class 1000 \
  --input_size 64 \
  --eval_step 1 \
  --view all \
  --save-ckpt

CUDA_VISIBLE_DEVICES=4 python downstream_classification.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained Pretrained_PATH \
  --finetune-dataset SLR \
  --protocol cross_subject \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based \
  --checkpoint-path SAVE_PATH \
  --optim SGD \
  --subset_name WLASL \
  --num_class 2000 \
  --input_size 64 \
  --eval_step 1 \
  --view all \
  --save-ckpt

CUDA_VISIBLE_DEVICES=4 python downstream_classification.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained Pretrained_PATH \
  --finetune-dataset SLR \
  --protocol cross_subject \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based \
  --checkpoint-path SAVE_PATH \
  --optim SGD \
  --subset_name NMFs_CSL \
  --num_class 1067 \
  --input_size 64 \
  --eval_step 1 \
  --view all \
  --save-ckpt

CUDA_VISIBLE_DEVICES=4 python downstream_classification.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained Pretrained_PATH \
  --finetune-dataset SLR \
  --protocol cross_subject \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based \
  --checkpoint-path SAVE_PATH \
  --optim SGD \
  --subset_name SLR \
  --num_class 500 \
  --input_size 64 \
  --eval_step 1 \
  --view all \
  --save-ckpt
