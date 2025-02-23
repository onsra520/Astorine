import os, joblib, sys
import pandas as pd
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from data.pipelines.transformation import datatransformer
print(project_root)
