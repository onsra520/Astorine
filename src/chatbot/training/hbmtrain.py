import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.chatbot.utils.ccol import load_features
from nlp.evaluator import hybridmodel

project_root = Path(__file__).resolve().parents[1]

paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "odata": os.path.abspath(f"{project_root}/data/storage/processed/final_cleaning.csv"),    
    "config": os.path.abspath(f"{project_root}/config/hybridmodel_config.json"),
    "models": os.path.abspath(f"{project_root}/models"),
    "encoders": os.path.abspath(f"{project_root}/models/encoders"),
    "scalers": os.path.abspath(f"{project_root}/models/scalers"),
}

os.makedirs(paths["encoders"], exist_ok=True)
os.makedirs(paths["scalers"], exist_ok=True)

class Hybrid_Training:
    def __init__(
        self,
        train: bool = True,
        epochs: int = 50,
        weights_only: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        embedding_dim: int = 8,
        patience: int = 10,
        min_delta: float = 0.001
    ) -> None:
        # Initialize parameters
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.embedding_dim = embedding_dim
        self.patience = patience
        self.min_delta = min_delta
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load and preprocess data
        self.dataset = pd.read_csv(paths["odata"])
        self.categorical_columns, self.numerical_columns = load_features(
            paths["config"],
            Numerical=["PRICE"],
            Remove=["HAS A TOUCH SCREEN", "PRICE"]
        )
        self.df = self.dataset[self.categorical_columns + self.numerical_columns].copy()
        
        # Transform data and split into train/val
        self.transform(weights_only=weights_only)
        
        # Load tensors for training and validation
        self.load_tensor()
        self.load_tensor_dataset()
        
        # Initialize model, optimizer, and criterion
        self.model = hybridmodel(self.categorical_columns, self.numerical_columns, embedding_dim=embedding_dim)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Train the model
        if train:
            self.model_train()

    def transform(self, weights_only: bool = False) -> None:
        """Preprocess data: encode categorical variables, calculate SCORE, and scale numerical features."""
        # Remove "DEVICE" from categorical columns (assumed present initially)
        self.categorical_columns.pop(self.categorical_columns.index("DEVICE"))

        # Encode categorical columns
        for column in self.categorical_columns:
            encoding_path = os.path.join(paths["encoders"], f"{column.lower().replace(' ', '_')}.pkl")
            label_encoder = LabelEncoder()
            if os.path.exists(encoding_path):
                encoded_values = torch.load(encoding_path, weights_only=weights_only)
                self.df[column] = encoded_values.transform(self.df[column])
            else:
                self.df[column] = label_encoder.fit_transform(self.df[column])
                torch.save(label_encoder, encoding_path)

        # Calculate target "SCORE"
        self.df["SCORE"] = 0.5 * self.df["CPU PASSMARK RESULT"] + 0.5 * self.df["GPU PASSMARK (G3D) RESULT"]

        # Scale numerical columns and SCORE
        scaler = StandardScaler()
        self.df[self.numerical_columns + ["SCORE"]] = scaler.fit_transform(
            self.df[self.numerical_columns + ["SCORE"]]
        )
        torch.save(scaler, os.path.join(paths["scalers"], "scaler.pkl"))

        # Extract feature and target data
        self.categorical_data = self.df[self.categorical_columns].values
        self.numerical_data = self.df.drop(columns=self.categorical_columns + ["DEVICE", "SCORE"]).values
        self.y_data = self.df["SCORE"].values

        # Split into training and validation sets
        self.cat_train, self.cat_val, self.num_train, self.num_val, self.y_train, self.y_val = train_test_split(
            self.categorical_data, self.numerical_data, self.y_data, test_size=0.2, random_state=42
        )

    def load_tensor(self) -> None:
        """Convert training and validation data into PyTorch tensors."""
        # Training tensors
        self.numerical_train = torch.tensor(self.num_train, dtype=torch.float)
        self.categorical_train = torch.tensor(self.cat_train, dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float)

        # Validation tensors
        self.numerical_val = torch.tensor(self.num_val, dtype=torch.float)
        self.categorical_val = torch.tensor(self.cat_val, dtype=torch.long)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float)

    def load_tensor_dataset(self) -> None:
        """Create TensorDatasets and DataLoaders for training and validation."""
        # Training dataset and dataloader
        self.train_dataset = TensorDataset(self.numerical_train, self.categorical_train, self.y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        # Validation dataset and dataloader
        self.val_dataset = TensorDataset(self.numerical_val, self.categorical_val, self.y_val)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def model_train(self) -> None:
        """Train the model with validation and early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for batch in self.train_loader:
                numerical_batch, categorical_batch, target_batch = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                outputs = self.model(numerical_batch, categorical_batch)
                loss = self.criterion(outputs.squeeze(), target_batch)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Training stopped due to NaN loss at epoch {epoch+1}")
                    return
                
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * numerical_batch.size(0)
            train_loss = running_loss / len(self.train_dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}")

            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    numerical_batch, categorical_batch, target_batch = [b.to(self.device) for b in batch]
                    outputs = self.model(numerical_batch, categorical_batch)
                    loss = self.criterion(outputs.squeeze(), target_batch)
                    val_running_loss += loss.item() * numerical_batch.size(0)
            val_loss = val_running_loss / len(self.val_dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(paths["models"], "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")