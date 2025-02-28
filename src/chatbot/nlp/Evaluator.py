import os
from pathlib import Path
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[1]
paths = {
    "encoders": os.path.abspath(f"{project_root}/models/encoders"),
    "scalers": os.path.abspath(f"{project_root}/models/scalers"),   
}

class hybridmodel(nn.Module):
    def __init__(self, categorical_columns: list, numerical_columns: list, embedding_dim=8):
        """
        Constructor for HybridModel.

        Args:
            categorical_columns (list): List of categorical columns to use
            numerical_columns (list): List of numerical columns to use
            embedding_dim (int, optional): Dimensionality of the embeddings. Defaults to 8.
        """
        super().__init__()
        self.categorical_columns = categorical_columns
        self.numerical_features = numerical_columns
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for column in self.categorical_columns:
            encoding_path = os.path.join(
                paths["encoders"], f"{column.lower().replace(' ', '_')}.pkl"
            )
            if os.path.exists(encoding_path):
                encoder = torch.load(encoding_path, weights_only=False)
                num_classes = len(encoder.classes_)
            else:
                num_classes = 10
            self.embeddings[column] = nn.Embedding(num_classes, embedding_dim)
            total_embedding_dim += embedding_dim

        fc_input_dim = total_embedding_dim + len(numerical_columns)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor):

        """
        Forward pass of the HybridModel.

        Args:
            numerical: tensor of numerical features, shape [batch_size, num_numerical]
            categorical: tensor of categorical ids, shape [batch_size, num_categorical]

        Returns:
            output: tensor of prediction, shape [batch_size, 1]
        """
        embedded_list = []
        for i, column in enumerate(self.categorical_columns):
            emb_layer = self.embeddings[column]
            col_ids = categorical[:, i]
            emb = emb_layer(col_ids)
            embedded_list.append(emb)

        cat_embeddings = torch.cat(embedded_list, dim=1)
        x = torch.cat([cat_embeddings, numerical], dim=1)
        return self.fc(x)
