import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse

from model import HypertensionPredictor
from dataset import load_and_clean_csv, split_three_group, df_to_tensor, HypertensionDataset

# 參數
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Hypertension Predictor")

    parser.add_argument("--test_meth_csv", type=str, required=True, help="Path to test methylation CSV")
    parser.add_argument("--test_clinic_csv", type=str, required=True, help="Path to test clinical CSV")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--save_dir", type=str, default="./predictions", help="Directory to save predictions CSV")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")

    return parser.parse_args()

# load model function
def load_trained_model(ckpt_path, device, hidden_dim=256):
    model = HypertensionPredictor(
        dna_dim=1,
        c1_dim=1,
        c2_dim=1,
        c3_dim=1,
        hidden_dim=hidden_dim
    ).to(device)

    print(f"Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)
    model.eval()
    return model

# inference function
def inference_all(model, loader, device="cpu", return_attn=False):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for dna_b, c1_b, c2_b, c3_b, _ in loader:
            dna_b, c1_b, c2_b, c3_b = dna_b.to(device), c1_b.to(device), c2_b.to(device), c3_b.to(device)
            if return_attn:
                pred_b, _ = model(dna_b, c1_b, c2_b, c3_b, return_attn=True)
            else:
                pred_b = model(dna_b, c1_b, c2_b, c3_b)
            all_preds.append(pred_b.cpu())

    return torch.cat(all_preds, dim=0)  # (N,1)

# 機率轉label
def logits_to_label(logits, threshold=0.5):
    prob = torch.sigmoid(logits)
    label = (prob > threshold).int()
    return prob.cpu(), label.cpu()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # load test data
    dna_test = load_and_clean_csv(args.test_meth_csv)
    clinic_test = load_and_clean_csv(args.test_clinic_csv)
    user_id = dna_test['CaseNo'].tolist()

    # prepare testset
    cy_test, c1_test, c2_test, c3_test = split_three_group(clinic_test)
    dna_test, c1_test, c2_test, c3_test, cy_test = df_to_tensor(dna_test, c1_test, c2_test, c3_test, cy_test)

    test_dataset = HypertensionDataset(dna_test, c1_test, c2_test, c3_test, cy_test)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    model = load_trained_model(args.ckpt_path, args.device, hidden_dim=args.hidden_dim)

    # inference
    preds = inference_all(model, test_loader, device=args.device)
    prob, label = logits_to_label(preds)

    # save to CSV
    prob_np  = prob.numpy().reshape(-1)
    label_np = label.numpy().reshape(-1)
    user_id_np = user_id if isinstance(user_id, list) else user_id.cpu().numpy().reshape(-1)

    df = pd.DataFrame({
        "user_id": user_id_np,
        "probability": prob_np,
        "prediction_label": label_np
    })

    out_path = os.path.join(args.save_dir, "hypertension_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"CSV 已輸出： {out_path}")

if __name__ == "__main__":
    main()
