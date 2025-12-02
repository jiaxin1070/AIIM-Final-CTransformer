import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# attention weight of DNA
def plot_cpg_attention_per_layer(attn_weights_dict, clinical_names=["c1", "c2", "c3"], save_dir = "./checkpoints2"):
    """
    attn_weights_dict:
        {
          "c1": {"f1": [Tensor(B1,q1,k), Tensor(B2,q2,k), ...],
                 "f2": [...],
                 "f3": [...]},
          "c2": {...},
          "c3": {...}
        }
    """
    for layer in clinical_names:
        for feat in ["f1", "f2", "f3"]:
            weights_list = attn_weights_dict[layer][feat]
            if len(weights_list) == 0:
                continue

            # 每個 Tensor: (B_i, q_i, k)
            # 先找最大的 q_len
            max_q = max(w.shape[1] for w in weights_list)
            k_len = weights_list[0].shape[2]

            upsampled = []
            for w in weights_list:
                B_i, q_i, k_i = w.shape
                assert k_i == k_len

                if q_i != max_q:
                    # interpolate 需要 (B, C, L)，這裡把 q 當 L
                    w_ = w.permute(0, 2, 1)  # (B, k, q)
                    w_ = F.interpolate(w_, size=max_q, mode="linear", align_corners=False)
                    w_ = w_.permute(0, 2, 1)  # 回到 (B, max_q, k)
                else:
                    w_ = w

                upsampled.append(w_)

            # 重點：不同 batch 的 B_i 不同 → 用 cat 拼在 batch 維度
            all_weights = torch.cat(upsampled, dim=0)  # (sum_B, max_q, k_len)

            # 對所有 sample 平均 → (max_q, k_len)
            avg_weights = all_weights.mean(dim=0)      # (max_q, k_len)

            importance = avg_weights.mean(dim=1).detach().cpu().numpy()  # 對 key 維或 query 維取平均都可以，看你要畫什麼

            plt.figure(figsize=(10, 4))
            plt.plot(importance)
            plt.title(f"{layer} - {feat} attention importance")
            plt.xlabel("Query position")
            plt.ylabel("Avg attention")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"{layer} - {feat} attention importance"))


def map_query_to_dna_index(q_idx, feat_name, L_dna):
    """
    把不同層(f1,f2,f3)的 query index 對回「大約」對應的原始 DNA 位置。

    q_idx: 1D array-like, 例如 np.arange(q_len)
    feat_name: "f1", "f2", "f3"
    L_dna: 原始 DNA 長度 (ex: 5517)
    """
    level_stride = {"f1": 2, "f2": 4, "f3": 8}  # 依據 UNetEncoder stride 設定
    s = level_stride[feat_name]

    q_idx = np.asarray(q_idx)
    dna_pos = q_idx * s          # 粗略映射回原始 index
    dna_pos = np.clip(dna_pos, 0, L_dna - 1)  # 防止越界
    return dna_pos

# attention weight (原本的DNA位置)
def plot_dna_attention_per_layer(attn_weights_dict, L_dna, clinical_names=("c1", "c2", "c3"), save_dir = "./checkpoints2"):
    """
    對 attn_all 畫出:
    - 每個 clinical group (c1,c2,c3)
    - 在每一層 (f1,f2,f3)
    對 DNA 位置的平均注意力 (attention importance)。

    參數:
    ----------
    attn_weights_dict: dict
        形狀應該像:
        {
          "c1": {"f1": [Tensor(B1,q1,k), Tensor(B2,q2,k), ...],
                 "f2": [...],
                 "f3": [...]},
          "c2": {...},
          "c3": {...}
        }

    L_dna: int
        原始 DNA 序列長度 (例如 dna_train.shape[-1])

    clinical_names: list 或 tuple
        要畫的 clinical group key, 預設為 ("c1","c2","c3")
    """

    for layer in clinical_names:
        for feat in ["f1", "f2", "f3"]:
            weights_list = attn_weights_dict[layer][feat]
            if len(weights_list) == 0:
                print(f"[Warning] {layer}-{feat} 沒有收集到 attention，跳過。")
                continue

            # 每個 w: (B_i, q_i, k)
            # 先找最大的 query 長度，方便對齊
            max_q = max(w.shape[1] for w in weights_list)
            k_len = weights_list[0].shape[2]

            upsampled = []
            for w in weights_list:
                B_i, q_i, k_i = w.shape
                assert k_i == k_len, "不同 batch 的 key 維度 k_len 不一致，請檢查模型輸出。"

                if q_i != max_q:
                    # MultiheadAttention 的 attn_weights: (B, q, k)
                    # interpolate 需要 (B, C, L)，把 q 當作 L 來補長度
                    w_ = w.permute(0, 2, 1)  # (B, k, q)
                    w_ = F.interpolate(w_, size=max_q, mode="linear", align_corners=False)
                    w_ = w_.permute(0, 2, 1)  # 回到 (B, max_q, k)
                else:
                    w_ = w

                upsampled.append(w_)

            # 不同 batch 的 B_i 不同，用 cat 接起來
            # all_weights: (sum_B, max_q, k_len)
            all_weights = torch.cat(upsampled, dim=0)

            # 對所有 sample 平均 → (max_q, k_len)
            avg_weights = all_weights.mean(dim=0)  # (max_q, k_len)

            # 再對 key 維度平均，得到每個 query 位置的「重要性」 → (max_q,)
            importance = avg_weights.mean(dim=1).detach().cpu().numpy()

            # 建立 query index，並映射回原始 DNA 位置
            q_idx = np.arange(len(importance))  # 0 ~ max_q-1
            dna_pos = map_query_to_dna_index(q_idx, feat, L_dna)

            # 畫圖
            plt.figure(figsize=(10, 4))
            plt.plot(dna_pos, importance)
            plt.title(f"Attention over DNA positions | {layer}-{feat}")
            plt.xlabel("DNA position (approx original index)")
            plt.ylabel("Average attention")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"DNA | {layer} - {feat} attention importance"))


def average_attention_list(attn_list):
    """
    attn_list: list of tensors OR single tensor, each tensor shape (num_heads, L, L)
    return: (L, L) average over all tensors and heads
    """
    # 如果是單一 tensor，包成 list
    if isinstance(attn_list, torch.Tensor):
        attn_list = [attn_list]
    elif isinstance(attn_list, list):
        # flatten list in case of nested lists
        flat_list = []
        for x in attn_list:
            if isinstance(x, torch.Tensor):
                flat_list.append(x)
            elif isinstance(x, list):
                flat_list.extend(x)
        attn_list = flat_list
    else:
        raise TypeError(f"attn_list must be Tensor or list of Tensors, got {type(attn_list)}")

    # stack tensors
    stacked = torch.stack(attn_list, dim=0).float()  # shape: (num_tensors, num_heads, L, L)
    avg = stacked.mean(dim=0).mean(dim=0)  # mean over heads and tensors
    return avg  # shape: (L, L)

def plot_attention_grid(attn_weights,save_dir = "./checkpoints2"):
    fig, axes = plt.subplots(3, 3, figsize=(20, 20)

    Cs = ['c1', 'c2', 'c3']
    Fs = ['f1', 'f2', 'f3']

    for i, C in enumerate(Cs):
        for j, F in enumerate(Fs):
            ax = axes[i, j]
            try:
                attn_list = attn_weights[C][F]
                attn_avg = average_attention_list(attn_list)
                # 將橫軸設為 DNA (query)，縱軸設為臨床資料 (key)
                attn_avg = attn_avg.T
                sns.heatmap(attn_avg.cpu().detach().numpy(), cmap='viridis', ax=ax)
                ax.set_title(f'{C.upper()} {F.upper()}')
                ax.set_xlabel('DNA features')       # query
                ax.set_ylabel('Clinical features')  # key

            except KeyError:
                print(f'Warning: {C} {F} not found in attn_weights')
                ax.axis('off')

    #plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "DNA-Clinic heatmap"))

