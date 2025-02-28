import json5
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nlp.chatbot import CustomNN, DialogueDataset
from src.chatbot.nlp.nlp_utils import tokenize, stem, bag_of_words

class chatbot_training():
    def __init__(
        self,
        intent_dir: str,
        save_dir: str,
        hidden_size: int = 16,
        batch_size: int = 8,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        num_epochs: int = 500,
        mode: str = None,
    ) -> None:
        """
        Initializes the chatbot training class with configurations for model training.

        Args:
            intent_dir (str): Path to the directory containing the intents data.
            save_dir (str): Directory where the trained model will be saved.
            hidden_size (int, optional): Number of neurons in the hidden layer of the neural network. Defaults to 16.
            batch_size (int, optional): Number of samples per batch for training. Defaults to 8.
            dropout_rate (float, optional): Dropout rate for regularization in the neural network. Defaults to 0.3.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            num_epochs (int, optional): Number of training iterations over the entire dataset. Defaults to 500.
            mode (str, optional): Mode of operation, if "train", the model will be trained. Defaults to None.

        Attributes:
            intents_data (dict): Loaded intents data from the specified directory.
            input_size (int): Size of the input layer determined by the training data.
            output_size (int): Number of output classes determined by the tags.
            train_loader (DataLoader): DataLoader instance for handling training data batches.
            device (torch.device): Device on which the model is trained (GPU or CPU).
            model (CustomNN): Initialized neural network model ready for training or evaluation.
        """

        self.intents_data = json5.load(open(intent_dir, "r", encoding="utf-8"))
        X_train, Y_train = self.prepare_dataset()
        self.input_size = len(X_train[0])
        self.output_size = len(self.tags)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        dataset = DialogueDataset(X_train, Y_train)
        self.train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomNN(
            self.input_size, 
            self.hidden_size, 
            self.output_size, 
            self.dropout_rate,
        ).to(self.device)
        if mode == "train":
            self.train(save_dir)

    def prepare_dataset(self):
        """
        Prepares the training dataset by processing intents data to extract and tokenize patterns.
        
        This function populates the `all_words` and `tags` attributes with unique stemmed words 
        and tags, respectively, from the intents data. It creates a list of tuples `xy` containing 
        tokenized words and their corresponding tags.

        The function constructs the training data (`X_train`, `Y_train`) where each input is a 
        bag-of-words representation of a pattern sentence, and each output is the index of the 
        corresponding tag.

        Returns:
            tuple: Two numpy arrays, `X_train` containing bag-of-words vectors, and `Y_train` 
            containing integer labels representing the tag indices.
        """
        self.all_words = []
        self.tags = []
        xy = []  

        ignore_words = ['?', '!', '.', ',']

        for intent in self.intents_data['intents']:
            if "tag" in intent and "patterns" in intent:
                tag = intent['tag']
                self.tags.append(tag)
                for pattern in intent['patterns']:
                    words = tokenize(pattern)
                    self.all_words.extend(words)
                    xy.append((words, tag))

        self.all_words = sorted(set([stem(word) for word in self.all_words if word not in ignore_words]))

        X_train = []
        Y_train = []
        for (pattern_sentence, tag) in xy:
            bag = bag_of_words(pattern_sentence, self.all_words)
            X_train.append(bag)
            label = self.tags.index(tag)
            Y_train.append(label)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train

    def train(self, save_dir: str):
        """
        Trains the model using the training data and saves the trained model.

        Args:
            save_dir (str): Directory path to save the trained model.

        The function uses CrossEntropyLoss as the criterion and Adam optimizer. It iterates over
        the number of epochs specified, processing batches of data from the train_loader. For each
        batch, it performs a forward pass to compute outputs, calculates the loss, and applies
        backpropagation to update the model parameters. The loss is printed every 100 epochs.
        Finally, the trained model and associated metadata (input size, hidden size, output size,
        vocabulary, etc.) are saved to the specified directory.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for (words, labels) in self.train_loader:
                words = words.clone().detach().to(self.device)
                labels = labels.clone().detach().to(self.device)
                outputs = self.model(words)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        print(f'Final loss: {loss.item():.4f}')
        
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }
        torch.save(data, save_dir)