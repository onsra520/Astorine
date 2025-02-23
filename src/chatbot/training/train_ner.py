import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.classify_columns import load_features
from nlp.Evaluator import HybridModel

base_dir = Path(__file__).resolve().parents[1]

paths = {
    "processed": os.path.abspath(f"{base_dir}/data/storage/processed"),
    "config": os.path.abspath(f"{base_dir}/config/config.json"),
    "models": os.path.abspath(f"{base_dir}/models"),
    "encoders": os.path.abspath(f"{base_dir}/models/encoders"),
    "scalers": os.path.abspath(f"{base_dir}/models/scalers"),
}

os.makedirs(paths["encoders"], exist_ok=True)
os.makedirs(paths["scalers"], exist_ok=True)

dataset = pd.read_csv(os.path.join(paths["processed"], "final_cleaning.csv"))
categorical_columns, numerical_columns = load_features(
    paths["config"], Remove=["HAS A TOUCH SCREEN"]
)
df = dataset[categorical_columns + numerical_columns].copy()
categorical_columns.pop(categorical_columns.index("DEVICE"))

for column in categorical_columns:
    encoding_path = os.path.join(
        paths["encoders"], f"{column.lower().replace(' ', '_')}.pkl"
    )
    label_encoder = LabelEncoder()
    if os.path.exists(encoding_path):
        encoded_values = torch.load(encoding_path, weights_only=False)
        df[column] = encoded_values.transform(df[column])
    else:
        df[column] = label_encoder.fit_transform(df[column])
        torch.save(label_encoder, encoding_path)

df["SCORE"] = 0.5 * df["CPU PASSMARK RESULT"] + 0.5 * df["GPU PASSMARK (G3D) RESULT"]

scaler = StandardScaler()
df[numerical_columns + ["SCORE"]] = scaler.fit_transform(
    df[numerical_columns + ["SCORE"]]
)
torch.save(scaler, os.path.join(paths["scalers"], "scaler.pkl"))

categorical_data = df[categorical_columns].values
numerical_data = df.drop(columns=categorical_columns + ["DEVICE", "SCORE"]).values
y_data = df["SCORE"].values

numerical_tensor = torch.tensor(numerical_data, dtype=torch.float)
categorical_tensor = torch.tensor(categorical_data, dtype=torch.long)
y_tensor = torch.tensor(y_data, dtype=torch.float)

dataset_tensor = TensorDataset(numerical_tensor, categorical_tensor, y_tensor)
dataloader = DataLoader(dataset_tensor, batch_size=32, shuffle=True)

model = HybridModel(categorical_columns, numerical_columns, embedding_dim=8)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 50
model.train()

def model_train():
    """
    Train the ComparingModel using the given DataLoader.

    Args:

    Returns:
    """
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            numerical_batch, categorical_batch, target_batch = batch
            optimizer.zero_grad()
            outputs = model(
                numerical_batch, categorical_batch
            )
            loss = criterion(outputs.squeeze(), target_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * numerical_batch.size(0)
        epoch_loss = running_loss / len(dataset_tensor)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        if epoch_loss < 0.01:
            print("Early stopping: loss < 0.01")
            break

    torch.save(model.state_dict(), os.path.join(paths["models"], "laptop_evaluator.pth"))

if __name__ == "__main__":
    model_train()
