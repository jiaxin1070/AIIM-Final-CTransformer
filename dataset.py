import pandas as pd
import torch
from torch.utils.data import Dataset



# 讀檔
def load_and_clean_csv(path, drop_cols=None):
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    return df


# 拆分
def split_three_group(data):
    g1 = ['CaseNo', 'SEX', 'age_b', 'edu', 'smoking', 'DRK', 'betel', 'SPORT',
          'weight_b', 'HEIGHT_b', 'BMI_b', 'WHR_b', 'T_CHO_b']

    g2 = ['CaseNo','TG_b', 'HDL_b', 'LDL_b', 'HBA1C_b', 'FBG_b', 'T_BILIRUBIN_b',
            'albumin_b', 'SGOT_b', 'SGPT_b', 'GAMMA_GT_b', 'BUN_b', 'creatinine_b', 'uric_acid_b']

    g3 = ['CaseNo','microalbumin_b', 'egfr_b', 'SBP_b', 'DBP_b', 'HR_b',
            'SBP', 'DBP', 'HR', 'HTN_b', 'HTN', 'HTN_FAM']
    
    label = data["HTN"]
    c1 = data[g1]
    c2 = data[g2]
    c3 = data[g3]

    return label, c1, c2, c3


def df_to_tensor(dna, c1, c2, c3, cy):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = dna.shape[0]

    # 先把 DataFrame 轉成全 float
    dna = dna.apply(pd.to_numeric, errors="coerce").fillna(0)
    c1  = c1.apply(pd.to_numeric, errors="coerce").fillna(0)
    c2  = c2.apply(pd.to_numeric, errors="coerce").fillna(0)
    c3  = c3.apply(pd.to_numeric, errors="coerce").fillna(0)

    dna = torch.tensor(dna.values, dtype=torch.float32).unsqueeze(1).to(device)
    c1  = torch.tensor(c1.values,  dtype=torch.float32).unsqueeze(1).to(device)
    c2  = torch.tensor(c2.values,  dtype=torch.float32).unsqueeze(1).to(device)
    c3  = torch.tensor(c3.values,  dtype=torch.float32).unsqueeze(1).to(device)
    cy = torch.tensor(cy.values, dtype=torch.float32).view(-1, 1).to(device)

    return dna, c1, c2, c3, cy


# 定義dataset
class HypertensionDataset(Dataset):
    def __init__(self, dna, c1, c2, c3, labels):
        """
        dna, c1, c2, c3, labels 都是 Tensor
        形狀可以是:
          - features: (N, L)  或  (N, C, L)
          - labels:  (N,) 或 (N, 1)
        """
        self.dna = dna
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.labels = labels

        n = len(labels)
        assert len(self.dna) == n
        assert len(self.c1) == n
        assert len(self.c2) == n
        assert len(self.c3) == n

    def __len__(self):
        return len(self.labels)

    def _ensure_channel_first(self, x):
        """
        確保每個 sample 是 (C, L)
        - 若 x.shape == (L,)      → 變成 (1, L)
        - 若 x.shape == (L1, L2)  → 當成 (C, L) 用（例如已經是 (1, L)）
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)        # (L,) → (1, L)
        return x

    def __getitem__(self, idx):
        dna = self._ensure_channel_first(self.dna[idx])
        c1  = self._ensure_channel_first(self.c1[idx])
        c2  = self._ensure_channel_first(self.c2[idx])
        c3  = self._ensure_channel_first(self.c3[idx])

        y = self.labels[idx]
        y = y.float()
        if y.ndim == 0:       # scalar → (1,)
            y = y.unsqueeze(0)

        return dna, c1, c2, c3, y




