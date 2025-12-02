import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
import sys
import os

def evaluate_classification(gt_csv, pred_csv, save_dir="./"):
    """
    gt_csv: str, path to ground truth CSV, 必須有 'user_id' 與 'label'
    pred_csv: str, path to prediction CSV, 必須有 'user_id' 與 'probability'、'prediction_label'
    save_dir: str, ROC curve 儲存目錄
    """
    os.makedirs(save_dir, exist_ok=True)

    # 讀取 CSV
    gt = pd.read_csv(gt_csv)
    gt = gt.rename(columns={"CaseNo": "user_id"})  # 根據你的 GT CSV 修改
    pred = pd.read_csv(pred_csv)

    # 對齊 user_id
    df = pd.merge(gt, pred, on="user_id", how="inner")

    y_true = df['HTN'].values
    y_prob = df['probability'].values
    y_pred = df['prediction_label'].values

    # 計算 AUROC
    auroc = roc_auc_score(y_true, y_prob)

    # 計算 Accuracy
    acc = accuracy_score(y_true, y_pred)

    # 計算 F1-score
    f1 = f1_score(y_true, y_pred)

    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")

    # 畫 ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.show()

    return auroc, acc, f1

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <gt_csv> <pred_csv> <save_dir>")
        sys.exit(1)

    gt_csv = sys.argv[1]
    pred_csv = sys.argv[2]
    save_dir = sys.argv[3]

    evaluate_classification(gt_csv, pred_csv, save_dir)
