# AIIM-Final-CTransformer

## train
```
python3 train.py \
    --train_meth_csv <dna train data path> \ # 換成自己的路徑
    --valid_meth_csv <dna valid data path> \
    --train_clinic_csv <clinical train data path> \
    --valid_clinic_csv <clinical valid data path> \
    --save_dir <path to save checkpoint>' \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 20 \
    --early_stop 3 \
    --hidden_dim 256
```

## inference
```
python3 inference.py \
    --test_meth_csv <dna test data path>  \ # 換成自己的路徑
    --test_clinic_csv <clinical test data path>  \
    --ckpt_path <checkpoint path>  \
    --save_dir <path to svae inference result> \
    --batch_size 1
```

## score
```
python3 score.py <ground truth path> <inference output csv path> <path to save score result> # 換成自己的路徑
```
