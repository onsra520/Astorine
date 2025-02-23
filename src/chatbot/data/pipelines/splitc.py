import os
import sys
import json5
import time 
import itertools
import pandas as pd
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

pd.set_option("future.no_silent_downcasting", True)

base_dir = Path(__file__).resolve().parents[2]
paths = {
    "raw": os.path.abspath(f"{base_dir}/data/storage/raw"),
    "processed": os.path.abspath(f"{base_dir}/data/storage/processed"),
    "config": os.path.abspath(f"{base_dir}/config/config.json"),
}

os.makedirs(paths["processed"], exist_ok=True)

def removemissing():
    """
    This function removes missing devices from the product dataset 
    if there are any missing devices in the product dataset.
    """
    List = []
    Spinner = itertools.cycle(["|", "/", "-", "\\"])
    print("Finding Missing Devices in Product Dataset: ", end="")

    File_Checked = [
        file for file in os.listdir(paths["raw"]) if file != "Device URLs.csv"
    ]

    for Original_File in File_Checked:
        for Comparing_File in File_Checked:
            print(
                f"\rFinding Missing Devices in Product Dataset: {next(Spinner)}",
                end="",
                flush=True,
            )
            File = pd.read_csv(os.path.join(paths["raw"], Original_File)).to_dict(
                "list"
            )["DEVICE"]
            Comparing = pd.read_csv(os.path.join(paths["raw"], Comparing_File)).to_dict(
                "list"
            )["DEVICE"]
            Missing_Device = set(File).symmetric_difference(set(Comparing))
            List += Missing_Device

    if len(List) != 0:
        Delete_List = pd.Series(List).drop_duplicates().tolist()
        print(f"\n{len(Delete_List)} Missing Devices found. Deleting...", end="")

        for File in File_Checked:
            print(f"\r{' ' * 50}\rDeleting from {File}", end="", flush=True)
            Delete_File = pd.read_csv(os.path.join(paths["raw"], File))
            Deleted_File = Delete_File[~Delete_File["DEVICE"].isin(Delete_List)]
            Deleted_File.to_csv(
                os.path.join(paths["raw"], File),
                header=True,
                index=False,
                encoding="utf-8",
            )
            time.sleep(0.5)

        print("\nAll missing devices have been deleted. Done!")
    else:
        print("\nNo Missing Devices Found!")

class splitdata:
    def __init__(self, config_path=paths["config"], raw_data_path=paths["raw"]):
        self.config = self._load_config(config_path)
        self.raw_data_path = raw_data_path
        self._load_datasets()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and return configuration from a JSON file.

        Parameters:
        config_path (str): The path to the JSON configuration file.

        Returns:
        Dict[str, Any]: The configuration data loaded from the JSON file.
        """
        with open(config_path, "r") as f:
            return json5.load(f)

    def _load_datasets(self) -> None:
        """
        Load all datasets specified in the configuration file.

        This method reads each dataset file specified in the configuration
        file and assigns it to the corresponding attribute in the class
        instance. The datasets are loaded as pandas DataFrames from the
        CSV files located in the raw data path.

        Parameters:
        None

        Returns:
        None
        """
        data_config = self.config["data_files"]
        for attribute_name, file_info in data_config.items():
            file_path = os.path.join(self.raw_data_path, file_info["file"])
            df = pd.read_csv(file_path)
            setattr(self, attribute_name, df)

    def __str__(self) -> str:
        return f"DeviceDataProcessor with {len(self.__dict__)-2} loaded datasets"

    def __repr__(self) -> str:
        return self.__str__()

    def _add_column_prefix(self, df: pd.DataFrame, prefix: str, exclude: str) -> pd.DataFrame:
        """
        Add a prefix to each column name in the DataFrame, except for the specified column to exclude.

        Parameters:
        df (pd.DataFrame): The DataFrame whose column names will be modified.
        prefix (str): The prefix to add to the column names.
        exclude (str): The name of the column to exclude from having the prefix added.

        Returns:
        pd.DataFrame: A new DataFrame with the updated column names.
        """
        return df.rename(
            columns={col: f"{prefix} {col}" for col in df.columns if col != exclude}
        )

    def _filter_na_columns(self, df: pd.DataFrame, max_na_ratio: float) -> pd.DataFrame:
        """Remove columns with NA ratio exceeding threshold"""
        na_ratios = df.isna().mean()
        valid_columns = na_ratios[na_ratios <= max_na_ratio].index.tolist()
        return df[valid_columns]

    def split_performance_data(self) -> pd.DataFrame:
        """Process and merge all performance-related datasets"""
        cpu_df = self._add_column_prefix(self.CPU_Performance, "CPU", exclude="DEVICE")

        # Process GPU data
        gpu_columns = self.config["data_files"]["GPU_Performance"]["columns"]
        gpu_df = self._add_column_prefix(self.GPU_Performance[gpu_columns], "GPU", exclude="DEVICE")

        # Process main performance data
        performance_df = self._filter_na_columns(self.Performance, max_na_ratio=0.75)
        columns_to_drop = self.config["data_files"]["Performance"]["columns"]
        performance_df = performance_df.drop(columns=columns_to_drop)

        # Merge datasets
        merged = pd.merge(performance_df, cpu_df, on="DEVICE", how="outer")
        merged = pd.merge(merged, gpu_df, on="DEVICE", how="outer")

        # Final cleanup
        merged = merged.rename(columns={"VRAM": "VRAM OF GPU"})
        return merged.fillna("Undefined")

    def split_device_information(self) -> pd.DataFrame:
        """Process and merge device specification datasets"""
        design_columns = self.config["data_files"]["Design"]["columns"]
        design_df = self.Design[design_columns]

        # Process connectivity data
        connectivity_df = self._filter_na_columns(self.Connectivity, max_na_ratio=0.75)
        connectivity_df = connectivity_df.fillna("Undefined")

        display_columns = self.config["data_files"]["Display"]["columns"]
        display_df = self.Display[display_columns]

        battery_columns = self.config["data_files"]["Battery"]["columns"]
        battery_df = self.Battery[battery_columns]

        # Process connectivity data
        connectivity_columns = self.config["data_files"]["Connectivity"]["columns"]
        connectivity_df = self._filter_na_columns(self.Connectivity, max_na_ratio=0.75)
        connectivity_df = connectivity_df.fillna("Undefined")
        connectivity_df = connectivity_df.drop(columns=connectivity_columns)

        # Merge datasets
        info_df = pd.merge(design_df, display_df, on="DEVICE", how="outer")
        info_df = pd.merge(info_df, battery_df, on="DEVICE", how="outer")
        info_df = pd.merge(info_df, connectivity_df, on="DEVICE", how="outer")

        # Add derived columns
        info_df.insert(0, "BRAND", info_df["DEVICE"].str.split().str[0])
        info_df["PRICE"] = "No Price Found"
        return info_df.fillna("Undefined")

if __name__ == "__main__":
    removemissing()
    splitdata()