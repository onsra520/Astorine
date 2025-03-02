import os, sys
from pathlib import Path
import pandas as pd
import torch

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from nlp.extractor import filters
from training.hbmtrain import Hybrid_Training

paths = {
    "hbm": os.path.abspath(f"{root}/models/compararison.pth"),
}

cheapest =  [
    "cheap",
    "cheapest",
    "least expensive",
    "most affordable",
    "inexpensive",
    "low cost",
    "budget friendly",
    "economical",
    "cost effective",
    "rock bottom",
    "lowest priced",
    "value priced",
    "discounted",
    "thrifty",
    "cut-rate",
    "dirt cheap",
    "bargain-basement",
    "affordable",
    "wallet friendly",
    "low priced"
]

hbm = Hybrid_Training(train=False)
model = hbm.model
model.load_state_dict(
    torch.load(paths["hbm"], weights_only=False)
)
model.eval()

def searching(query: str=None) -> pd.DataFrame:
    sorted_df = hbm.df.copy()
    if sorted_df.empty:
        return sorted_df
    
    numerical_input = torch.tensor(sorted_df[hbm.numerical_columns].values, dtype=torch.float, device=hbm.device)
    categorical_input = torch.tensor(sorted_df[hbm.categorical_columns].values, dtype=torch.long, device=hbm.device)

    model.eval()
    with torch.no_grad():
        outputs = model(numerical_input, categorical_input)
        predicted_scores = outputs.squeeze().cpu().numpy()
    
    sorted_df["PREDICTED_SCORE"] = predicted_scores
    sorted_filtered = sorted_df.sort_values(
        by=["PREDICTED_SCORE", "CPU PASSMARK RESULT", "GPU PASSMARK (G3D) RESULT"],
        ascending=[False, False, False]
    )
    sorted_filtered.columns = [col.lower() for col in sorted_filtered.columns]
    if query is None:
        return sorted_filtered["device"]
    
    extract_text = filters(query=str(query))    
    result = pd.merge(sorted_filtered, extract_text, on='device', how='right', sort=False)

    query_lower = query.lower()
    if any(txt in query_lower for txt in cheapest):
        list_of = result.sort_values(by="price", ascending=True)
    else:
        list_of = result.sort_values(by="predicted_score", ascending=False)
    return list_of[["device", "price", "predicted_score"]]["device"].head(5).unique().tolist()