import os, sys
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent))
from handlers.responses import reply
from handlers.rcm import searching
from nlp.helper.ibuilder import igenerate_lite
from nlp.extractor import extract
from decimal import Decimal
import random

model_dir = os.path.join(os.path.join(Path(__file__).resolve().parent), "models\\chatbotmodel.pth" )
intents_dir = os.path.join(os.path.join(Path(__file__).resolve().parent), "intents\\intents_lite.json" )
data =  igenerate_lite(save=False)

sessions = {}

required_fields = [
    "brand", "gpu", "cpu", "ram", "resolution", "refresh rate",
    "display type", "screen size", "use_for", "price"
]

price_responses = [
    "💰 What is your budget for this purchase?",
    "💵 Could you please let me know your spending limit?",
    "🤑 How much money are you planning to invest?",
    "💸 May I know your budget range?",
    "🎯 What amount have you set aside for this?",
    "🔍 Can you share the budget you have in mind?"
]

work_responses = [
    "💼 Wonderful! What will you primarily use this laptop for?",
    "🔍 Great! Could you tell me the main purpose of your work?",
    "🚀 Excellent! What tasks or work will the laptop be used for?"
]

class ChatbotSession:
    def __init__(self):
        self.current_flow = "Nothing"  # "guided", "search", hoặc "faq"
        self.context = None            # Ví dụ: "price" khi hỏi về giá
        self.criteria = {              # Lưu trữ tiêu chí
            "brand": None,
            "gpu": None,
            "cpu": None,
            "ram": None,
            "resolution": None,
            "refresh rate": None,
            "display type": None,
            "screen size": None,
            "use_for": None,
            "price": {'min': Decimal('0'), 'max': Decimal('0')}
        }
        self.previous_flow = None
    
    def reset(self):
        self.current_flow = "Nothing"  # "guided", "search", hoặc "faq"
        self.context = None            # Ví dụ: "price" khi hỏi về giá
        self.criteria = {              # Lưu trữ tiêu chí
            "brand": None,
            "gpu": None,
            "cpu": None,
            "ram": None,
            "resolution": None,
            "refresh rate": None,
            "display type": None,
            "screen size": None,
            "use_for": None,
            "price": {'min': Decimal('0'), 'max': Decimal('0')}
        }
        self.previous_flow = None

def get_session(user_id):
    if user_id not in sessions:
        sessions[user_id] = ChatbotSession()
    return sessions[user_id]

def update_session(session: ChatbotSession, user_input: str):
    extracted = extract(user_input)
    for key, value in extracted.items():
        if value is not None:
            session.criteria[key] = value

    criteria_count = len([key for key, value in session.criteria.items() if value is not None])
    if criteria_count >= 3:
        session.current_flow = "search"
        if session.criteria.get("price") == {'min': Decimal('0'), 'max': Decimal('0')}:
            session.context = "price"
            return random.choice(price_responses)
        else:
            try:
                results = searching(user_input)
                if  results:
                    formatted_results = "\n".join([f"{i+1}. {laptop}" for i, laptop in enumerate(results)])
                    session.reset()
                    return f"Here are the laptops I found:\n{formatted_results}\nI think they are good for you."
                else:
                    return "Sorry, no laptops match your criteria."
            except:
                return "Sorry, I couldn't find any laptops matching your criteria."
            
    tag_response = reply(user_input)
    tag = tag_response.get("tag")
    response = tag_response.get("response")
    
    if tag == "help":
        session.current_flow = "guided"
        session.context = None
        session.criteria = {field: None for field in required_fields}
        session.criteria["price"] = {'min': Decimal('0'), 'max': Decimal('0')}
        return response 
    
    if tag == "use_for":
        session.criteria["use_for"] = extracted.get("use_for", user_input)
        session.context = "price"
        return random.choice(price_responses)
    
    if session.context == "price":
        if extracted.get("price") is not None and extracted["price"] != {'min': Decimal('0'), 'max': Decimal('0')}:
            session.criteria["price"] = extracted["price"]
            criteria_string = " | ".join(str(session.criteria.get(field) or "") for field in required_fields)
            results = searching(criteria_string)
            session.current_flow = "search"
            if results:
                formatted_results = "\n".join([f"{i+1}. {laptop}" for i, laptop in enumerate(results)])
                session.reset()
                return f"Here are the laptops I found:\n{formatted_results}\nI think they are good for you."
            else:
                return "I couldn't find any laptops matching your criteria."
        return random.choice(price_responses)
    
    faq_tags = ["gpu_question", "cpu_question", "ram_question"]
    if session.current_flow == "guided" and tag in faq_tags:
        faq_response = response
        resume_message = f"{faq_response} So, what {tag.split('_')[0]} do you prefer?"
        return resume_message
    
    if session.current_flow == "guided":
        for field in required_fields:
            if session.criteria.get(field) is None:
                if field == "use_for":
                    return random.choice(work_responses)
                elif field == "price":
                    return random.choice(price_responses)
                else:
                    return response
                
        criteria_string = " | ".join(str(session.criteria.get(field) or "") for field in required_fields)
        results = searching(criteria_string)
        session.current_flow = "search"
        if results:
            formatted_results = "\n".join([f"{i+1}. {laptop}" for i, laptop in enumerate(results)])
            session.reset()
            return f"Here are the laptops I found:\n{formatted_results}\nI think they are good for you."
        else:
            return "I couldn't find any laptops matching your criteria."
    return response

def chatbot_handle(user_id: str, user_input: str):
    session = get_session(user_id)
    response = update_session(session, user_input)
    return response

if __name__ == "__main__":
    while True:
        user_id = "user_dep_trai_nhat_qua_dat"
        user_input = input("User: ")
        if user_input == "bye":
            print(chatbot_handle(user_id, user_input))
            break
        print(chatbot_handle(user_id, user_input))