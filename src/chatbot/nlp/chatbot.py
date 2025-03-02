import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.utils.prune as prune

class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super(CustomNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2) 
        self.ln2 = nn.LayerNorm(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size) 
        self.ln3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.l4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        identity = out
        out = self.l2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.l3(out)
        out = self.ln3(out)
        out = self.relu(out)
        if hasattr(self, 'input_size') and self.input_size == self.hidden_size:
            out = out + identity 
        out = self.dropout3(out)

        out = self.l4(out)
        return out

    def optimize(self, pruning_amount=0.3):
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
        half_model = self.half()
        try:
            scripted_model = torch.jit.script(half_model)
            return scripted_model
        except Exception as e:
            print(f"Error during scripting: {e}")
            return half_model
    
    def __getstate__(self):
        state = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'state_dict': self.state_dict()
        }
        return state

    def __setstate__(self, state):
        self.__init__(state['input_size'], state['hidden_size'], state['num_classes'], state['dropout_rate'])
        self.load_state_dict(state['state_dict'])

class DialogueDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.n_samples = len(X_train)
        self.x_data = torch.FloatTensor(np.array(X_train))
        self.y_data = torch.LongTensor(np.array(Y_train))

            
        if isinstance(Y_train, np.ndarray):
            self.y_data = torch.LongTensor(Y_train)
        else:
            self.y_data = torch.LongTensor(np.array(Y_train))
            
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.n_samples