import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.utils.prune as prune

class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        """
        Initializes the CustomNN model with three linear layers, layer normalization, dropout, and ReLU activation.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layers.
            num_classes (int): The number of output classes.
            dropout_rate (float, optional): The dropout rate to apply after each hidden layer. Default is 0.3.
        """
        super(CustomNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        The forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        out = self.l1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.l2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.l3(out)
        return out

    def optimize(self, pruning_amount=0.3):
        """
        Optimizes the neural network model by applying pruning, converting to half precision, 
        and scripting for deployment.

        This method performs the following optimizations:
        1. Prunes the weights of Linear layers using L1 unstructured pruning based on the given pruning amount.
        2. Converts the model to half precision to reduce memory footprint and improve inference speed.
        3. Uses TorchScript to script the model for optimized deployment.

        Args:
            pruning_amount (float): The proportion of weights to prune in each Linear layer.

        Returns:
            torch.jit.ScriptModule: The scripted version of the optimized model.
        """
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
        self.half()
        scripted_model = torch.jit.script(self)
        return scripted_model
    
    def __getstate__(self):
        """
        Returns a dictionary containing the hyperparameters and state_dict of the model.
        
        This method is used to serialize the essential components of the model, 
        allowing it to be saved and later restored. The returned state includes 
        the input size, hidden size, number of classes, dropout rate, and the 
        model's state_dict.
        """

        state = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'state_dict': self.state_dict()
        }
        return state

    def __setstate__(self, state):
        """
        Reinitializes the model from a saved state.

        This method is used to deserialize the model from a saved state, 
        restoring the input size, hidden size, number of classes, dropout rate, and the 
        model's state_dict. The state is expected to be a dictionary containing the 
        essential components of the model. The model is reinitialized with the saved 
        hyperparameters and the weights are loaded from the state_dict.
        """
        self.__init__(state['input_size'], state['hidden_size'], state['num_classes'], state['dropout_rate'])
        self.load_state_dict(state['state_dict'])
        

class DialogueDataset(Dataset):
    def __init__(self, X_train, Y_train):
        """
        Initializes the DialogueDataset class with the training data.

        Args:
            X_train (list): List of feature vectors for the training data.
            Y_train (list): List of labels corresponding to the training data.

        Attributes:
            n_samples (int): Number of training samples.
            x_data (list): List of feature vectors for the training data.
            y_data (list): List of labels corresponding to the training data.
        """
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    def __getitem__(self, index):
        """
        Returns the feature vector and label at a given index.

        Args:
            index (int): The index of the feature vector and label to return.

        Returns:
            tuple: A tuple containing the feature vector and label at the given index.
        """
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        This method allows the use of the len() function to obtain the 
        number of feature-label pairs available in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.n_samples