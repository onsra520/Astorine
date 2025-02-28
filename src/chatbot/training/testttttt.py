import sys, os, json5
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import pipeline
from torch.utils.data import DataLoader, Dataset
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.chatbot.nlp.nlp_utils import tokenize, stem, bag_of_words
from src.chatbot.nlp.helper.ncomp import rlst, srlst, clst, glst, rrlst, dtlst, sslst, blst
from handlers.rcm import searching

project_root = Path(__file__).resolve().parents[1]

paths = {
    "intents": os.path.abspath(f"{project_root}/intents"),
    "patterns": os.path.abspath(f"{project_root}/intents/patterns.json"),
    "responses": os.path.abspath(f"{project_root}/intents/responses.json"),
    "replymodel": os.path.abspath(f"{project_root}/models/replymodel.pth"),
}

sessions = {}

class ChatbotSession:
    def __init__(self):
        self.current_flow = "guided"  # 'guided' hoặc 'search'
        self.context = None           # ví dụ: "Need_help", "collecting_criteria", v.v.
        self.criteria = {}            # lưu trữ các tiêu chí: brand, gpu, cpu, ...
        self.previous_flow = None     # lưu flow tạm thời khi trả lời FAQ

def get_session(user_id):
    if user_id not in sessions:
        sessions[user_id] = ChatbotSession()
    return sessions[user_id]