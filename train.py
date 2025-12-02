import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from model import HypertensionPredictor
from dataset import load_and_clean_csv, split_three_group, df_to_tensor, HypertensionDataset
from attention_weight_plot import plot_cpg_attention_per_layer, plot_dna_attention_per_layer, plot_attention_grid

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hypertension Predictor")

    # 資料路徑
    parser.add_argument("--train_meth_csv", type=str, required=True, help="Path to train methylation CSV")
    parser.add_argument("--valid_meth_csv", type=str, required=True, help="Path to valid methylation CSV")
    parser.add_argument("--train_clinic_csv", type=str, required=True, help="Path to train clinical CSV")
    parser.add_argument("--valid_clinic_csv", type=str, required=True, help="Path to valid clinical CSV")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints and plots")

    # 訓練參數
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--early_stop", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    return parser.parse_args()

def main():
    args = parse_args()

    # setup
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path   = os.path.join(args.save_dir, "best_model.pt")
    latest_ckpt_path = os.path.join(args.save_dir, "latest_model.pt")

    device = args.device

    # prepare dataset
    dna_train = load_and_clean_csv(args.train_meth_csv)
    dna_val   = load_and_clean_csv(args.valid_meth_csv)

    clinic_train = load_and_clean_csv(args.train_clinic_csv)
    clinic_val = load_and_clean_csv(args.valid_clinic_csv)

    cy_train, c1_train, c2_train, c3_train = split_three_group(clinic_train)
    cy_val, c1_val, c2_val, c3_val = split_three_group(clinic_val)

    dna_train, c1_train, c2_train, c3_train, cy_train = df_to_tensor(dna_train, c1_train, c2_train, c3_train, cy_train)
    dna_val, c1_val, c2_val, c3_val, cy_val = df_to_tensor(dna_val, c1_val, c2_val, c3_val, cy_val)

    train_dataset = HypertensionDataset(dna_train, c1_train, c2_train, c3_train, cy_train)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = HypertensionDataset(dna_val, c1_val, c2_val, c3_val, cy_val)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # initialize Model
    model = HypertensionPredictor(
        dna_dim=1,
        c1_dim=1,
        c2_dim=1,
        c3_dim=1,
        hidden_dim=args.hidden_dim
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # resume training if latest checkpoint exists
    if os.path.exists(latest_ckpt_path):
        print("Loading latest checkpoint...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # train
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for dna_b, c1_b, c2_b, c3_b, y_b in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            dna_b, c1_b, c2_b, c3_b, y_b = dna_b.to(device), c1_b.to(device), c2_b.to(device), c3_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(dna_b, c1_b, c2_b, c3_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            total_samples += len(y_b)
            total_correct += ((pred > 0).float() == y_b).sum().item()

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # validation
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        attn_all = {l:{f:[] for f in ["f1","f2","f3"]} for l in ["c1","c2","c3"]}

        with torch.no_grad():
            for dna_b, c1_b, c2_b, c3_b, y_b in val_loader:
                dna_b, c1_b, c2_b, c3_b, y_b = dna_b.to(device), c1_b.to(device), c2_b.to(device), c3_b.to(device), y_b.to(device)
                pred, attn = model(dna_b, c1_b, c2_b, c3_b, return_attn=True)
                loss = criterion(pred, y_b)

                val_loss += loss.item() * len(y_b)
                val_samples += len(y_b)
                val_correct += ((pred > 0.5).float() == y_b).sum().item()

                for layer in ["c1","c2","c3"]:
                    for tgt in ["f1","f2","f3"]:
                        attn_all[layer][tgt].append(attn[layer][tgt])

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        scheduler.step()

        # record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, latest_ckpt_path)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_ckpt_path)
            print("Saved BEST checkpoint!")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. ({patience_counter}/{args.early_stop})")

        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save loss & acc curves
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))

    plt.figure(figsize=(8,6))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "acc_curve.png"))
    print("Saved training curves")

    # attention heatmaps
    plot_cpg_attention_per_layer(attn_all, clinical_names=["c1","c2","c3"], save_dir=args.save_dir)
    plot_dna_attention_per_layer(attn_all, L_dna=dna_train.shape[-1], clinical_names=("c1","c2","c3"), save_dir=args.save_dir)

    # small sample attention grid
    dna_small = dna_train[:5].to(device)
    c1_small  = c1_train[:5].to(device)
    c2_small  = c2_train[:5].to(device)
    c3_small  = c3_train[:5].to(device)

    with torch.no_grad():
        pred, attn_weights = model(dna_small, c1_small, c2_small, c3_small, return_attn=True)

    plot_attention_grid(attn_weights, save_dir=args.save_dir)
    print("Saved attention plots")

if __name__ == "__main__":
    main()
