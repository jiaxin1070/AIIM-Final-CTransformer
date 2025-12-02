# AIIM-Final-CTransformer

## train
```
python3 train.py \
    --train_meth_csv '/content/drive/MyDrive/final/fake_data/train_methylation.csv' \ # 換成自己的路徑
    --valid_meth_csv '/content/drive/MyDrive/final/fake_data/valid_methylation.csv' \
    --train_clinic_csv '/content/drive/MyDrive/final/fake_data/clinic_train.csv' \
    --valid_clinic_csv '/content/drive/MyDrive/final/fake_data/clinic_val.csv' \
    --save_dir '/content/drive/MyDrive/final/checkpoints2' \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 20 \
    --early_stop 3 \
    --hidden_dim 256
```

## inference
```
python3 inference.py \
    --test_meth_csv '/content/drive/MyDrive/final/fake_data/test_methylation.csv'  \ # 換成自己的路徑
    --test_clinic_csv '/content/drive/MyDrive/final/fake_data/clinic_test.csv'  \
    --ckpt_path '/content/drive/MyDrive/final/checkpoints2/best_model.pt'  \
    --save_dir '/content/drive/MyDrive/final/result2' \
    --batch_size 1
```

## score
```
python3 score.py '/content/drive/MyDrive/final/fake_data/clinic_test.csv' '/content/drive/MyDrive/final/result2/hypertension_predictions.csv' '/content/drive/MyDrive/final/result2' # 換成自己的路徑
```
# AIIM-Final-CTransformer
