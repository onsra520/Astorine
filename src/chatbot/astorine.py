import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import json5
import torch
import random
from nlp.chatbot import CustomNN
from src.chatbot.nlp.nlp_utils import tokenize, stem, bag_of_words
from training.train_intent import chatbot_training

project_root = str(Path(__file__).parent)
paths = {
    "intents": os.path.abspath(f"{project_root}/intents"),
    "intents": os.path.abspath(f"{project_root}/intents/intents.json"),
    "replymodel": os.path.abspath(f"{project_root}/models/replymodel.pth"),
}

class chatbot():
    def __init__(
        self,
        model_dir: str = paths["replymodel"],
        intent_dir: str = paths["intents"],
        retrain: bool = False,
        ) -> None:
        self.intents_data = json5.load(open(intent_dir, "r", encoding="utf-8"))
        self.responses_dict = {
            intent["tag"]: intent["responses"]
            for intent in self.intents_data["intents"]
            if "tag" in intent and "responses" in intent
        }
        self._load_model(model_dir = model_dir, intent_dir = intent_dir, retrain = retrain)

    def _load_model(self, model_dir: str, intent_dir: str, retrain: bool) -> None:
        if not os.path.exists(paths["replymodel"]) or retrain == True:
            _ = chatbot_training(
                mode="train",
                intent_dir=intent_dir,
                save_dir=model_dir
                )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(model_dir, map_location=self.device)
        self.model = CustomNN(data["input_size"], data["hidden_size"], data["output_size"]).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        self.all_words = data["all_words"]
        self.tags = data["tags"]

    def predict_intent(self, user_input) -> str:
        words = tokenize(user_input)
        stemmed_words = [stem(word) for word in words]
        bag = bag_of_words(stemmed_words, self.all_words)
        bag = torch.from_numpy(bag).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(bag)
            _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            return tag
        return "unknown"

    def get_response(self, tag: str):
        if tag in self.responses_dict:
            return random.choice(self.responses_dict[tag])
        else:
            return "i'm sorry, i don't understand that."

def reply(
    query: str, 
    retrain: bool = False, 
    intent_dir: str = paths["intents"],
    model_dir: str = paths["replymodel"],
    ) -> str:
    chatbot_name = "Astorine"
    chatbot_usage = chatbot(
        retrain = retrain, 
        intent_dir = intent_dir, 
        model_dir = model_dir)
    intent = chatbot_usage.predict_intent(query)
    response = chatbot_usage.get_response(intent)
    return f"{chatbot_name}: {response}"

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        print(reply(user_input))
        if chatbot().predict_intent(user_input) == "goodbye":
            break
