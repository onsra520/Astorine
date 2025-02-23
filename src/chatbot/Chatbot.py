import os, json, torch, random
from pathlib import Path
from chatbot.nlp.ChatModel import NeuralNetwork
from chatbot.utils.nlp_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(current_dir, "intents", "intents.json")
with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

if "chatbot.pth" not in os.listdir(
    os.path.join(current_dir, "models")
):
    print("Model not found, training model...")
    from training import train_intent

FILE = os.path.join(current_dir, "models", "chatbot.pth")
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Astorine"

def get_response(message):
    """
    Given a message, respond with a corresponding response from the intents.json file
    if the probability of the predicted tag is greater than 0.75. Otherwise, return "I do not understand..."

    Args:
        message (str): the message to respond to

    Returns:
        str: the response to the message
    """
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I do not understand..."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence in ["quit", "exit", "stop"]:
            print("See ya later!")
            break
        resp = get_response(sentence)
        print(resp)
