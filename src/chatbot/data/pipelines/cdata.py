import os
import re
import sys
import time 
import itertools
import threading
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data.pipelines.performance import performance_cleaning
from data.pipelines.design import design_cleaning
from utils.pdsp import relocate_column

pd.set_option("future.no_silent_downcasting", True)

base_dir = Path(__file__).resolve().parents[2]
paths = {
    "raw": os.path.abspath(f"{base_dir}/data/storage/raw"),
    "processed": os.path.abspath(f"{base_dir}/data/storage/processed"),
    "config": os.path.abspath(f"{base_dir}/config/config.json"),
}

os.makedirs(paths["processed"], exist_ok=True)

class data_cleaning:
    modified_columns = [
        {"column": "PRICE", "before": None, "after": "BRAND"},
        {"column": "WARRANTY PERIOD", "before": "DEVICE", "after": "TYPE"},
        {"column": "CPU SPEED", "before": "CPU THREADS", "after": None},
    ]

    def __init__(self):
        self.data = None
        self.processed_path = paths["processed"]
        self.config_path = paths["config"]
        self.initialize_data()

    def loading_spinner(self, stop_event: threading.Event, message: str):
        """Show a loading spinner while the function is running

        This function is typically used when another function is running in a
        separate thread and we want to show a loading spinner to the user until
        the other function completes.

        Parameters
        ----------
        stop_event : threading.Event
            An event that can be set to stop the loading spinner
        """
        spinner = itertools.cycle(["|", "/", "-", "\\"])
        while not stop_event.is_set():
            print(f"\r{message}: {next(spinner)}", end="", flush=True)
            time.sleep(0.2)
        print(f"\r{message}: Done!")

    def initialize_data(self):
        """Initialize and prepare base datasets

        If processed performance and design data are not found, this method will
        trigger the cleaning process for both datasets.
        """

        if "performance_cleaning.csv" not in os.listdir(self.processed_path):
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=self.loading_spinner,
                args=(
                    stop_event,
                    "Performance data not found. Cleaning performance data",
                ),
            )
            spinner_thread.start()
            performance_cleaning()
            stop_event.set()
            spinner_thread.join()

        if "design_cleaning.csv" not in os.listdir(self.processed_path):
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=self.loading_spinner,
                args=(
                    stop_event,
                    "Design data not found. Cleaning design data",
                ),
            )
            spinner_thread.start()
            design_cleaning()
            stop_event.set()
            spinner_thread.join()
        self.process_data()

    def process_data(self) -> pd.DataFrame:
        """
        Execute complete data processing pipeline

        This method will first merge the performance and design datasets, then
        process the name and price columns, and finally reorder the columns
        according to the data_config.json file. The processed data will be
        saved to a CSV file.
        """
        processing_steps = [
            self.merge_data,
            self.process_name,
            self.process_price,
            self.process_cpuspeed,
            self.process_gpu,
            self.reorder_column,
        ]
        for step in processing_steps:
            step()
        self.save_data()
        
        return self.data

    def merge_data(self) -> pd.DataFrame:
        """Merge design and performance datasets
        This method merges the design and performance datasets using the "DEVICE"
        column as the common key. The merged dataset is then saved to the
        instance variable "data".
        """
        design_data = pd.read_csv(
            os.path.join(self.processed_path, "design_cleaning.csv")
        )
        performance_data = pd.read_csv(
            os.path.join(self.processed_path, "performance_cleaning.csv")
        )
        data = pd.merge(design_data, performance_data, on="DEVICE", how="right")
        self.data = data.drop(columns=["BRAND_y"]).rename(columns={"BRAND_x": "BRAND"})

    def UpdateProductName(self, name: str):
        """Update product name by replacing SSD with " 512GB SSD" if the DEVICE
        column value is split by "/" and has a length of 4. If not, return "Undefined".
        """
        if len(name.split("/")) == 4:
            SSD = name.split("/")[-1]
            return name.replace(SSD, " 512GB SSD")
        else:
            return "Undefined"

    def process_name(self) -> pd.DataFrame:
        """
        Process name column by updating product name and replacing "-" in CPU column.
        Update product name by replacing SSD with " 512GB SSD" if the DEVICE column value
        is split by "/" and has a length of 4. If not, return "Undefined".
        Replace "-" in CPU column with " ".
        """
        self.data["DEVICE"] = self.data["DEVICE"].apply(
            lambda name: self.UpdateProductName(name)
        )
        self.data["CPU"] = self.data["CPU"].apply(lambda CPU: CPU.replace("-", " "))

    def process_price(self) -> pd.DataFrame:
        """
        Process price column by rounding to nearest 1000 and applying a special function to it.
        The special function is defined in the Generate_Price class in the supporter module.
        The function takes a price as input and returns an integer value close to the input price.
        The returned value is then multiplied by 16681, divided by 1000, and rounded to the
        nearest integer before being assigned back to the PRICE column in the dataframe.
        """
        self.data["PRICE"] = self.data.apply(
            lambda Price: self.Generate_Price(Price) , axis=1
        )

    def process_cpuspeed(self) -> pd.DataFrame:
        self.data.drop("CPU SPEED", axis=1, inplace=True)
        self.data["CPU SPEED"] = self.data["CPU CLOCK MULTIPLIER"].apply(
            lambda x: x * 100 / 1000
        )

    def process_gpu(self) -> pd.DataFrame:
        self.data["GPU"] = self.data["GPU"].apply(
            lambda x: x.replace("Nvidia GeForce 3060", "Nvidia GeForce RTX 3060")
            if x == "Nvidia GeForce 3060"
            else x
        )

    def reorder_column(self) -> pd.DataFrame:
        """Reorder dataframe columns"""
        for column in self.modified_columns:
            column_name = column["column"]
            if column["before"] or column["after"]:
                self.data = relocate_column(
                    Information=self.data,
                    Column_Name=column_name,
                    Before=column["before"],
                    Behind=column["after"],
                )

    def ASCII_Price(self, Name):
        """
        Calculate the price of a laptop based on the ASCII sum of its name.

        Parameters
        ----------
        Name : str
            The name of the laptop.

        Returns
        -------
        int
            The calculated price of the laptop.
        """
        Price = sum(ord(char) for char in Name)
        return Price

    def Generate_Price(self, Product):
        """
        Generate a price for a laptop based on its specifications.

        Parameters
        ----------
        Product : pandas Series
            A row of a dataframe containing information about the laptop,
            including its brand, CPU, GPU, RAM, screen size, and refresh rate, etc.

        Returns
        -------
        float
            The generated price of the laptop.
        """
        Base_Price = 100

        CPU = Product["CPU"].replace("AMD Ryzen", " ").replace("Intel Core", " ")
        CPU_Price = self.ASCII_Price(CPU)

        GPU = (
            Product["GPU"]
            .replace("Nvidia GeForce", " ")
            .replace("AMD Radeon", " ")
            .replace("Nvidia", " ")
            .replace("Max-Q", "9")
        )
        GPU_Price = self.ASCII_Price(GPU)
        RAM_Price = Product["RAM"] / 16 * 40
        Screen_Size_Price = int(Product["SCREEN SIZE"] - 13) * 50
        Refresh_Rate = int(Product["REFRESH RATE"]) / 100 * 35   
        Resolution = self.ASCII_Price(Product["RESOLUTION"]) / 10
        Weight = 40 if int(Product["WEIGHT"]) < 1.3 else (20 if int(Product["WEIGHT"]) < 1.9 else int(Product["WEIGHT"]))
        Base_Price += CPU_Price + GPU_Price + RAM_Price + Screen_Size_Price + Refresh_Rate + Resolution + Weight
        if Product["BRAND"] in ["Asus", "Dell", "LG", "Microsoft", "Razer", "Samsung"]:
            Base_Price *= 1.2
        elif Product["BRAND"] in ["Acer", "Gigabyte", "HP", "Lenovo", "MSI"]:
            Base_Price *= 1.1
        noise = np.random.uniform(-150, 150)
        final_price = Base_Price + noise
        return round(final_price, 2)

    def save_data(self):
        """Save the merged and cleaned data to a CSV file"""
        self.data.to_csv(
            os.path.join(self.processed_path, "final_cleaning.csv"), index=False
        )

if __name__ == "__main__":
    data_cleaning()
