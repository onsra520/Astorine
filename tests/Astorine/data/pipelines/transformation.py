import os, joblib, sys
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from supporter import load
from cleaning import cleaning

base_dir = Path(__file__).resolve().parents[1]
paths = {
    "processed": os.path.join(base_dir, "storage", "processed"),
    "config": os.path.join(base_dir, "config", "config.json"),
    "models": os.path.join(base_dir.parent, "models"),
    "encoders": os.path.join(base_dir.parent, "models", "encoders"),
}
os.makedirs(paths["encoders"], exist_ok=True)

# class datatransformer:
#     category_features = [
#         "BRAND", "DEVICE", "CPU", "GPU", "TYPE", "RESOLUTION",
#         "DISPLAY TYPE", "HAS A TOUCH SCREEN"
#         ]

#     numerical_features = [
#         "PRICE", "WEIGHT", "WIDTH", "HEIGHT", "THICKNESS", "SCREEN SIZE",'REFRESH RATE',
#         'BATTERY SIZE', 'RAM', 'RAM SPEED','CPU CORES', 'CPU THREADS', 'CPU SPEED',
#         'TURBO CLOCK SPEED','CPU CLOCK MULTIPLIER', 'VRAM OF GPU', 'GPU CLOCK SPEED',
#         'GPU TURBO', 'SEMICONDUCTOR SIZE', 'GPU MEMORY SPEED','GPU NUMBER OF TRANSISTORS',
#         'GPU THERMAL DESIGN POWER (TDP)', "GPU MEMORY BUS WIDTH", "GPU EFFECTIVE MEMORY SPEED",
#         "GPU MAXIMUM MEMORY BANDWIDTH", "GPU SHADING UNITS", "GPU TEXTURE RATE","GPU PIXEL RATE",
#         "GPU RENDER OUTPUT UNITS (ROPS)", "GPU TEXTURE MAPPING UNITS (TMUS)",
#         "GPU FLOATING-POINT PERFORMANCE", "CPU PASSMARK RESULT", "GPU PASSMARK (G3D) RESULT"
#         ]

#     def __init__(self):
#         """
#         Initialize transformer object.

#         This method will initialize the transformer object and prepare the base datasets by calling the initialize_data method.

#         Attributes:
#             dataset (pd.DataFrame): The transformed dataset.
#             dataframe (pd.DataFrame): The cleaned dataset before transformation.
#         """
#         self.cat = self.category_features
#         self.num = self.numerical_features
#         self.dataset = None
#         self.dataframe = None
#         self.initialize_data()

#     def initialize_data(self):
#         """
#         Initialize and prepare base datasets.

#         If processed data is not found, this method will trigger the cleaning process.
#         """
#         if "final_cleaning.csv" not in os.listdir(paths["processed"]):
#             print("Unpaid data, Cleaning is being done ...")
#             cleaning()
#         self.dataset = pd.read_csv(os.path.join(paths["processed"], "final_cleaning.csv"))
#         self.transform_data()
#         self.save_transformed_data()

#     def transform_data(self) -> None:
#         """
#         Transform the data by encoding categorical features and scaling numerical features.

#         Parameters
#         ----------
#         dataset : pd.DataFrame, optional
#             The dataset to be transformed. If None, the dataset in the object will be used.

#         Returns
#         -------
#         None
#         """
#         encoder_features, numerical_features = load(
#             JsonFilePath=paths["config"],
#             Categorical=self.cat,
#             Numerical=self.num,
#         )
#         self.dataframe = self.dataset[encoder_features + numerical_features].copy()

#         for feature in encoder_features:
#             if feature != "DEVICE":
#                 encoder = LabelEncoder()
#                 self.dataframe[f"{feature}_encoded"] = encoder.fit_transform(self.dataframe.loc[:, feature])
#                 self.dataframe.drop(feature, axis=1, inplace=True)
#                 joblib.dump(encoder, f"{paths["encoders"]}/{'_'.join(feature.lower().split(' '))}_encoder.pkl")

#         scaler = StandardScaler()
#         self.dataframe[numerical_features] = scaler.fit_transform(self.dataframe[numerical_features])

#     def save_transformed_data(self):
#         """Save the transformed data to a csv file."""
#         self.dataframe.to_csv(
#             os.path.join(paths["processed"], "trasformed_data.csv"), index=False
#         )


# if __name__ == "__main__":
#     datatransformer()
