import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data.pipelines.splitc import splitdata
from utils.pdsp import manage_column_list, find_similar_rows

pd.set_option("future.no_silent_downcasting", True)

base_dir = Path(__file__).resolve().parents[2]
paths = {
    "raw": os.path.abspath(f"{base_dir}/data/storage/raw"),
    "processed": os.path.abspath(f"{base_dir}/data/storage/processed"),
    "config": os.path.abspath(f"{base_dir}/config/config.json"),
}

os.makedirs(paths["processed"], exist_ok=True)

class design_cleaning:
    unnecessary_columns = [
        "HAS A BACKLIT KEYBOARD", "USB PORTS", "THUNDERBOLT PORTS", "HDMI VERSION",
        "DISPLAYPORT OUTPUTS", "HAS AN HDMI OUTPUT", "HAS USB TYPE-C", "DOWNLOAD SPEED",
        "UPLOAD SPEED", "MINI DISPLAYPORT OUTPUTS", "ETHERNET PORTS", "SUPPORTS WI-FI",
        "HAS AN EXTERNAL MEMORY SLOT", "BLUETOOTH VERSION", "HAS A VGA CONNECTOR",
        "USB 3.0 PORTS", "USB 2.0 PORTS", "USB 4 20GBPS PORTS", "THUNDERBOLT 3 PORTS",
        "USB 3.2 GEN 1 PORTS (USB-C)",
    ]

    reversed_columns = []

    def __init__(self):
        self.config_path = paths["config"]
        self.processed_path = paths["processed"]
        self.initialize_data()

    def initialize_data(self):
        """Initialize and prepare base datasets"""
        design_data = splitdata().split_device_information()
        self.design_data = design_data.drop(
            columns=self.get_clean_columns(), errors="ignore"
        )
        self.process_data()

    def get_clean_columns(self) -> List[str]:
        """Manage column configuration in JSON file"""
        if len(self.unnecessary_columns) != 0:
            for col in self.unnecessary_columns:
                manage_column_list(JsonFilePath=self.config_path, Remove=col)
        if len(self.reversed_columns) != 0:
            for col in self.reversed_columns:
                manage_column_list(JsonFilePath=self.config_path, Recovers=col)
        return manage_column_list(JsonFilePath=self.config_path)

    def save_data(self):
        """Save processed data to CSV"""
        self.design_data.to_csv(
            os.path.join(self.processed_path, "design_cleaning.csv"), index=False
        )

    def process_data(self):
        """Execute complete data processing pipeline"""
        processing_steps = [
            self.process_weight,
            self.process_appearance,
            self.process_warranty_period,
            self.process_type,
            self.process_battery,
            self.process_monitor,
            self.process_connectivity,
            self.save_data,
        ]

        for step in processing_steps:
            step()

    def process_weight(self) -> pd.DataFrame:
        # WEIGHT - Remove "kg" or "g" and set Undefined to similarity
        self.design_data["WEIGHT"] = (
            self.design_data.apply(
                lambda Weight: (
                    (
                        float(Weight["WEIGHT"].split(" ")[0])
                        if "kg" in Weight["WEIGHT"]
                        else (
                            float(Weight["WEIGHT"].split(" ")[0]) / 1000
                            if float(Weight["WEIGHT"].split(" ")[0]) > 600
                            else float(Weight["WEIGHT"].split(" ")[0]) / 100
                        )
                    )
                    if Weight["WEIGHT"] != "Undefined"
                    else find_similar_rows(
                        dataframe=self.design_data,
                        target_row=Weight,
                        reference_column="WEIGHT",
                        matching_columns="DEVICE",
                        comparison_column="DEVICE",
                        mode="similarity",
                        similarity_threshold=60,
                    )
                    .replace("kg", "")
                    .replace("g", "")
                    .strip()
                ),
                axis=1,
            )
        ).astype(float)

    def process_appearance(self) -> pd.DataFrame:
        # WIDTH, HEIGHT and THICKNESS- Convert "mm" to "cm" and set Undefined to similarity
        for Column in ["WIDTH", "HEIGHT", "THICKNESS"]:
            self.design_data[Column] = (
                self.design_data.apply(
                    lambda Size: (
                        round((float(Size[Column].replace("mm", "").strip()) / 100), 2)
                        if Size[Column] != "Undefined"
                        else round(
                            (
                                float(
                                    find_similar_rows(
                                        dataframe=self.design_data,
                                        target_row=Size,
                                        reference_column=Column,
                                        matching_columns="DEVICE",
                                        comparison_column="DEVICE",
                                        mode="similarity",
                                        similarity_threshold=60,
                                    )
                                    .replace("mm", "")
                                    .strip()
                                )
                                / 100
                            ),
                            2,
                        )
                    ),
                    axis=1,
                )
            ).astype(float)

    def process_warranty_period(self) -> pd.DataFrame:
        avg_period = [
            {"BRAND": "Acer", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Apple", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Asus", "WARRANTY PERIOD": "2 Years"},
            {"BRAND": "Casper", "WARRANTY PERIOD": "2 Years"},
            {"BRAND": "Dell", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Durabook", "WARRANTY PERIOD": "3 Years"},
            {"BRAND": "Gigabyte", "WARRANTY PERIOD": "2 Years"},
            {"BRAND": "HP", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "LG", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Lenovo", "WARRANTY PERIOD": "2 Years"},
            {"BRAND": "MSI", "WARRANTY PERIOD": "2 Years"},
            {"BRAND": "Microsoft", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Razer", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Samsung", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "XMG", "WARRANTY PERIOD": "2 Years"},
            {"BRAND": "XPG", "WARRANTY PERIOD": "1 Year"},
            {"BRAND": "Xiaomi", "WARRANTY PERIOD": "1 Year"},
        ]

        if "WARRANTY PERIOD" not in self.design_data.columns:
            self.design_data["WARRANTY PERIOD"] = "Undefined"

        self.design_data["WARRANTY PERIOD"] = self.design_data.apply(
            lambda Warranty: (
                next(
                    (
                        entry["WARRANTY PERIOD"]
                        for entry in avg_period
                        if entry["BRAND"] == Warranty["BRAND"]
                    ),
                    Warranty["WARRANTY PERIOD"],
                )
                if Warranty["WARRANTY PERIOD"] == "Undefined"
                or Warranty["WARRANTY PERIOD"] == "0 years"
                else Warranty["WARRANTY PERIOD"]
            )
            .split(" ")[0]
            .strip(),
            axis=1,
        )

    def process_type(self) -> pd.DataFrame:
        self.design_data["TYPE"] = np.where(
            self.design_data["WEIGHT"] < 1.9,
            "Gaming, Productivity, Ultrabook",
            "Gaming, Productivity",
        )

    def process_battery(self) -> pd.DataFrame:
        self.design_data["BATTERY SIZE"] = self.design_data.apply(
            lambda Battery: (
                Battery["BATTERY SIZE"].replace("Wh", "")
                if Battery["BATTERY SIZE"] != "Undefined"
                else find_similar_rows(
                    dataframe=self.design_data,
                    target_row=Battery,
                    reference_column="BATTERY SIZE",
                    matching_columns="DEVICE",
                    comparison_column="BRAND",
                    mode="similarity",
                    similarity_threshold=50,
                )
                .replace("Undefined", "90")
                .replace("Wh", "")
                .strip()
            ),
            axis=1,
        ).astype(float)

    def process_monitor(self) -> pd.DataFrame:
        # SCREEN SIZE - Remove "inch" and set Undefined to similarity
        self.design_data["SCREEN SIZE"] = self.design_data["SCREEN SIZE"].apply(
            lambda Screen: Screen.replace('"', "").strip()
        )

        # RESOLUTION - Remove "px" and set Undefined to similarity
        self.design_data["RESOLUTION"] = self.design_data.apply(
            lambda Resolution: (
                Resolution["RESOLUTION"]
                .replace("2400 x 0", "2400 x 1600")
                .replace("px", "")
                .strip()
                if Resolution["RESOLUTION"] != "Undefined"
                else find_similar_rows(
                    dataframe=self.design_data,
                    target_row=Resolution,
                    reference_column="RESOLUTION",
                    matching_columns="DEVICE",
                    comparison_column="DEVICE",
                    mode="similarity",
                    similarity_threshold=80,
                )
                .replace("px", "")
                .strip()
            ),
            axis=1,
        )

        # DISPLAY TYPE - Set Undefined to similarity
        self.design_data["DISPLAY TYPE"] = self.design_data.apply(
            lambda Display_Type: (
                Display_Type["DISPLAY TYPE"]
                if Display_Type["DISPLAY TYPE"] != "Undefined"
                else find_similar_rows(
                    dataframe=self.design_data,
                    target_row=Display_Type,
                    reference_column="DISPLAY TYPE",
                    matching_columns="DEVICE",
                    comparison_column="DEVICE",
                    mode="similarity",
                    similarity_threshold=75,
                ).replace("Undefined", "LED-backlit, IPS")
            ),
            axis=1,
        )

        # REFRESH RATE - Remove "Hz" and set Undefined to similarity
        self.design_data["REFRESH RATE"] = self.design_data.apply(
            lambda Refresh_Rate: (
                Refresh_Rate["REFRESH RATE"].replace("Hz", "").strip()
                if Refresh_Rate["REFRESH RATE"] not in ["Undefined", "0Hz"]
                else find_similar_rows(
                    dataframe=self.design_data,
                    target_row=Refresh_Rate,
                    reference_column="REFRESH RATE",
                    matching_columns=["BRAND", "DEVICE", "RESOLUTION"],
                    comparison_column="DEVICE",
                    mode="similarity",
                    similarity_threshold=70,
                )
                .replace("Undefined", "120")
                .replace("Hz", "")
                .strip()
            ),
            axis=1,
        ).astype(int)

    def process_connectivity(self) -> pd.DataFrame:
        # Danh sách các cột liên quan đến cổng USB
        self.process_usb_C()
        self.process_usb_A()
        self.process_other_ports()

    def process_usb_C(self) -> pd.DataFrame:
        usb_columns = [
            "USB 3.2 GEN 2 PORTS (USB-C)",
            "USB 4 40GBPS PORTS",
            "THUNDERBOLT 4 PORTS",
        ]

        for col in usb_columns:
            self.design_data[col] = self.design_data.apply(
                lambda row: (
                    row[col]
                    if row[col] != "Undefined"
                    else find_similar_rows(
                        dataframe=self.design_data,
                        target_row=row,
                        reference_column=col,
                        matching_columns="DEVICE",
                        comparison_column="DEVICE",
                        mode="similarity",
                        similarity_threshold=60,
                    ).replace("Undefined", "0")
                ),
                axis=1,
            )

        self.design_data["USB-C"] = self.design_data.apply(
            lambda row: (
                int(row["USB 3.2 GEN 2 PORTS (USB-C)"])
                + int(row["USB 4 40GBPS PORTS"])
                + int(row["THUNDERBOLT 4 PORTS"])
            ),
            axis=1,
        )

        self.design_data["USB-C"] = self.design_data["USB-C"].apply(
            lambda row: 2 if int(row) >= 3 else row
        )

        self.design_data.drop(columns=usb_columns, inplace=True)

    def process_usb_A(self) -> pd.DataFrame:
        usb_columns = ["USB 3.2 GEN 1 PORTS (USB-A)", "USB 3.2 GEN 2 PORTS (USB-A)"]

        for col in usb_columns:
            self.design_data[col] = self.design_data.apply(
                lambda row: (
                    row[col]
                    if row[col] != "Undefined"
                    else find_similar_rows(
                        dataframe=self.design_data,
                        target_row=row,
                        reference_column=col,
                        matching_columns="DEVICE",
                        comparison_column="DEVICE",
                        mode="similarity",
                        similarity_threshold=60,
                    ).replace("Undefined", "1")
                ),
                axis=1,
            )

        self.design_data["USB-A"] = self.design_data.apply(
            lambda row: (
                int(row["USB 3.2 GEN 1 PORTS (USB-A)"])
                + int(row["USB 3.2 GEN 2 PORTS (USB-A)"])
            ),
            axis=1,
        )
        self.design_data["USB-A"] = self.design_data["USB-A"].apply(
            lambda row: 2 if row in [0, 1] else row
        )
        self.design_data.drop(columns=usb_columns, inplace=True)

    def process_other_ports(self) -> pd.DataFrame:
        self.design_data["WI-FI VERSION"] = self.design_data["WI-FI VERSION"].replace(
            "Undefined", "Wi-Fi 6 (802.11ax), Wi-Fi 4 (802.11n), Wi-Fi 5 (802.11ac)"
        )

        self.design_data["RJ45 PORTS"] = self.design_data["RJ45 PORTS"].replace(
            "Undefined", "1"
        )

        self.design_data["HDMI PORTS"] = self.design_data["HDMI PORTS"].replace(
            "Undefined", "1"
        )

if __name__ == "__main__":
    design_cleaning()
