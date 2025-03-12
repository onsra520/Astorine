import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import random
from nlp.chatbot import CustomNN
from nlp.nlp_utils import *
from training.cbtrain import chatbottraining
from nlp.helper.ibuilder import igenerate_lite

class chatbot():
    hidden_size: int = 64
    batch_size: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 3e-4
    num_epochs: int = 200
    validation_split: float = 0.2
    patience: int = 10
    use_cuda: bool = True

    def __init__(
        self,
        path: str,
        training_data: dict,
        train: bool = False,
        fine_tune: bool = False,
        hidden_size: int = None,
        batch_size: int = None,
        dropout_rate: float = None,
        learning_rate: float = None,
        num_epochs: int = None,
        validation_split: float = None,
        patience: int = None,
        use_cuda: bool = True
        ) -> None:

        self.hidden_size = hidden_size if hidden_size is not None else chatbot.hidden_size
        self.batch_size = batch_size if batch_size is not None else chatbot.batch_size
        self.dropout_rate = dropout_rate if dropout_rate is not None else chatbot.dropout_rate
        self.learning_rate = learning_rate if learning_rate is not None else chatbot.learning_rate
        self.num_epochs = num_epochs if num_epochs is not None else chatbot.num_epochs
        self.validation_split = validation_split if validation_split is not None else chatbot.validation_split
        self.patience = patience if patience is not None else chatbot.patience
        self.use_cuda = use_cuda if use_cuda is not None else chatbot.use_cuda

        if train == True and training_data and path:
            _ = chatbottraining(
                path=path,
                training_data=training_data,
                hidden_size=self.hidden_size,
                batch_size=self.batch_size,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
                num_epochs=self.num_epochs,
                validation_split=0.2,
                patience=15
            )

        if fine_tune and training_data and path:
            self._fine_tune_model(path, training_data)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.intents_data = training_data
        self.responses_dict = {
            intent["tag"]: intent["responses"]
            for intent in self.intents_data["intents"]
            if "tag" in intent and "responses" in intent
        }
        self._load_model(path = path)

    def _fine_tune_model(self, path, training_data):
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        _ = chatbottraining(
            path=path,
            training_data=training_data,
            hidden_size=checkpoint["hidden_size"],
            batch_size=self.batch_size,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate / 10,
            num_epochs=self.num_epochs // 2,
            validation_split=self.validation_split,
            patience=self.patience,
            use_cuda=self.use_cuda
        )

    def _load_model(self, path: str) -> None:
        """Load the trained model with error handling."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = CustomNN(
                checkpoint["input_size"],
                checkpoint["hidden_size"],
                checkpoint["output_size"],
                dropout_rate=self.dropout_rate
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self.all_words = checkpoint["all_words"]
            self.tags = checkpoint["tags"]
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")

    def predict_intent(self, query: str = None) -> str:
        if not query:
            return "unknown"

        words = tokenize(query)
        stemmed_words = [stem(word) for word in words]
        bag = bag_of_words(stemmed_words, self.all_words)
        bag = torch.from_numpy(bag).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model(bag)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, dim=1)

        tag_idx = predicted.item()
        prob = probs[0][tag_idx].item()
        top_probs, _ = torch.topk(probs, 2, dim=1)
        top_prob_diff = top_probs[0][0].item() - top_probs[0][1].item()
        confidence_threshold = 0.6 if top_prob_diff > 0.3 else 0.75

        if prob > confidence_threshold:
            return self.tags[tag_idx]
        return "unknown"

    def get_response(self, tag: str):
        if tag in self.responses_dict:
            responses = self.responses_dict[tag]
            if len(responses) > 1:
                response_weights = [len(r) for r in responses]
                total_weight = sum(response_weights)
                probabilities = [w/total_weight for w in response_weights]
                return np.random.choice(responses, p=probabilities)
            return random.choice(responses)
        else:
            return "I'm sorry, I don't understand that."

model_dir = os.path.join(str(Path(__file__).resolve().parents[1]), "models\\chatbotmodel.pth" )
intents_dir = os.path.join(str(Path(__file__).resolve().parents[1]), "intents\\intents_lite.json" )
data = igenerate_lite(save=True, save_dir=intents_dir)

def reply(query: str, path: str = model_dir, training_data: dict= data) -> str:
    chat = chatbot(
        path = path,
        training_data = training_data
    )
    tag = chat.predict_intent(query)
    response = chat.get_response(tag)
    intents = {
        "tag":tag,
        "response": f"{response}"
    }
    return intents
