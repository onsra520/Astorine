import os
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data.pipelines.splitc import splitdata
from utils.pdsp import relocate_column, manage_column_list, find_similar_rows

pd.set_option("future.no_silent_downcasting", True)

base_dir = Path(__file__).resolve().parents[2]
paths = {
    "raw": os.path.abspath(f"{base_dir}/data/storage/raw"),
    "processed": os.path.abspath(f"{base_dir}/data/storage/processed"),
    "config": os.path.abspath(f"{base_dir}/config/config.json"),
}

os.makedirs(paths["processed"], exist_ok=True)

class performance_cleaning():
    """Class for cleaning and processing performance-related device data"""
    unnecessary_columns = [
        "MEMORY BANDWIDTH", "CPU GEEKBENCH 5 RESULT (MULTI)", "CPU GEEKBENCH 5 RESULT (SINGLE)",
        "CPU CINEBENCH R20 (MULTI) RESULT", "CPU CINEBENCH R20 (SINGLE) RESULT", "USB 3.0 PORTS",
        "USB PORTS", "THUNDERBOLT PORTS", "HDMI VERSION", "UPLOAD SPEED", "ETHERNET PORTS",
        "DISPLAYPORT OUTPUTS", "DOWNLOAD SPEED", "USB 2.0 PORTS", "MINI DISPLAYPORT OUTPUTS",
    ]

    modified_columns = [
        {"column": "BRAND", "rename": None, "before": None, "after": "DEVICE"},
        {"column": "DDR MEMORY VERSION", "rename": None, "before": "RAM", "after": None},
        {"column": "CPU", "rename": None, "before": "RAM SPEED", "after": None},
        {"column": "CPU THREADS", "rename": None, "before": "CPU", "after": None},
        {"column": "CPU SPEED", "rename": None, "before": "CPU THREADS", "after": None},
        {"column": "TURBO CLOCK SPEED", "rename": None, "before": "CPU SPEED", "after": None},
        {"column": "CPU CORES", "rename": None, "before": None, "after": "CPU THREADS"},
        {"column": "GPU CLOCK MULTIPLIER", "rename": "CPU CLOCK MULTIPLIER", "before": "TURBO CLOCK SPEED", "after": None},
        {"column": "USES MULTITHREADING", "rename": None, "before": "CPU CLOCK MULTIPLIER", "after": None},
        {"column": "GPU PASSMARK (G3D) RESULT", "rename": None, "before": "GPU GEEKBENCH 6 RESULT (SINGLE)", "after": None},
        {"column": "GPU GPU MEMORY SPEED", "rename": "GPU MEMORY SPEED", "before": None, "after": None},
        {"column": "GPU GEEKBENCH 6 RESULT (SINGLE)", "rename": "CPU GEEKBENCH 6 RESULT (SINGLE)", "before": None, "after": None},
        {"column": "GPU GEEKBENCH 6 RESULT (MULTI)", "rename": "CPU GEEKBENCH 6 RESULT (MULTI)", "before": None, "after": None},
        {"column": "CPU PASSMARK RESULT", "rename": None, "before": "GPU FLOATING-POINT PERFORMANCE", "after": "CPU PASSMARK RESULT (SINGLE)"},
        {"column": "GPU", "rename": None, "before": "USES MULTITHREADING", "after": None},
    ]
    gpu_struc = [
        {"name": "SEMICONDUCTOR SIZE", "type": "independent", "slitword": "nm", "matching": None, "comparison": None, "mode": None, "threshold": None},
        {"name": "VRAM OF GPU", "type": "independent", "slitword": "GB", "matching": None, "comparison": None, "mode": None, "threshold": None},
        {"name": "PCI EXPRESS (PCIE) VERSION", "type": "independent", "slitword": " ", "matching": None, "comparison": None, "mode": None, "threshold": None},
        {"name": "GPU GPU MEMORY SPEED", "type": "independent", "slitword": "MHz", "matching": None, "comparison": None, "mode": None, "threshold": None},
        {"name": "GPU GDDR VERSION", "type": "independent", "slitword": " ", "matching": None, "comparison": None, "mode": None, "threshold": None},
        {"name": "GPU TURBO", "type": "dependent", "slitword": "MHz", "matching": "GPU", "comparison": "GPU", "mode": "similarity", "threshold": 95},
        {"name": "GPU CLOCK SPEED", "type": "dependent", "slitword": "MHz", "matching": "GPU", "comparison": "GPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU THERMAL DESIGN POWER (TDP)", "type": "dependent", "slitword": "W", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU MEMORY BUS WIDTH", "type": "dependent", "slitword": "-", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU EFFECTIVE MEMORY SPEED", "type": "dependent", "slitword": "MHz", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU MAXIMUM MEMORY BANDWIDTH", "type": "dependent", "slitword": "GB/s", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU SHADING UNITS", "type": "dependent", "slitword": " ", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU TEXTURE RATE", "type": "dependent", "slitword": "GTexels/s", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU PIXEL RATE", "type": "dependent", "slitword": "GPixel/s", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU RENDER OUTPUT UNITS (ROPS)", "type": "dependent", "slitword": " ", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU TEXTURE MAPPING UNITS (TMUS)", "type": "dependent", "slitword": " ", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU FLOATING-POINT PERFORMANCE", "type": "dependent", "slitword": "TFLOPS", "matching": "CPU", "comparison": "CPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU NUMBER OF TRANSISTORS", "type": "dependent", "slitword": "million", "matching": "GPU", "comparison": "GPU", "mode": "similarity", "threshold": 90},
        {"name": "GPU PASSMARK (G3D) RESULT", "type": "dependent", "slitword": " ", "matching": "GPU", "comparison": "GPU", "mode": "similarity", "threshold": 85},
    ]

    def __init__(self):
        self.config_path = paths["config"]
        self.processed_path = paths["processed"]
        self.initialize_data()

    def initialize_data(self):
        """
        Initialize and prepare base datasets
        
        If processed performance and design data are not found, this method will
        trigger the cleaning process for both datasets.
        """
        performance_data = splitdata().split_performance_data()
        self.data = performance_data.drop(columns=self.get_clean_columns(), errors="ignore")
        self.process_data()

    def get_clean_columns(self) -> List[str]:
        """
        Manage column configuration in JSON file by removing unnecessary columns.

        Iterates over the `unnecessary_columns` list and removes each column from the
        configuration JSON file specified by `self.config_path`. The updated list of columns
        that are deemed unnecessary is then returned.

        Returns:
            List[str]: A list of columns that are not needed in data processing.
        """
        if len(self.unnecessary_columns) != 0:
            for col in self.unnecessary_columns:
                manage_column_list(JsonFilePath=self.config_path, Remove=col)
        return manage_column_list(JsonFilePath=self.config_path)

    def process_data(self):
        """Execute complete data processing pipeline"""
        processing_steps = [
            self.clean_brand_data,
            self.process_memory,
            self.process_cpu,
            self.process_gpu,
            self.reorder_column,
            self.final_processing,
        ]

        for step in processing_steps:
            step()

    # Brand processing methods
    def clean_brand_data(self):
        """Clean and organize brand-related data"""
        self.data["BRAND"] = self.data["DEVICE"].str.split().str[0]
        self.filter_brand("Apple")

    def filter_brand(self, brand: str):
        """Filter out specific brand from dataset"""
        self.data = self.data[self.data["BRAND"] != brand]

    # Memory processing methods
    def process_memory(self):
        """Process memory-related features"""
        self.clean_ram_data()
        self.process_ddr_version()
        self.process_ram_speed()

    def clean_ram_data(self):
        """Clean RAM-related columns"""
        self.data["RAM"] = self.data["RAM"].str.replace("GB", "")

    def process_ddr_version(self):
        """Process DDR memory version information"""
        self.data["DDR MEMORY VERSION"] = (
            self.data["DDR MEMORY VERSION"].str.split().str[0]
        )
        current_years = ["2023", "2024"]

        ddr_conditions = [
            self.data["DDR MEMORY VERSION"].str.contains(
                "|".join(current_years), na=False
            ),
            self.data["DDR MEMORY VERSION"] == "Undefined",
        ]

        ddr_values = ["5", "4"]

        self.data["DDR MEMORY VERSION"] = np.select(
            ddr_conditions, ddr_values, default=self.data["DDR MEMORY VERSION"]
        )

    def process_ram_speed(self):
        """Calculate RAM speed based on DDR version"""
        base_speeds = {"5": "4800", "4": "3200", "Undefined": "3200", "0": "3200"}

        self.data["RAM SPEED"] = self.data["DDR MEMORY VERSION"].map(base_speeds)
        self.data["RAM SPEED"] = self.data["RAM SPEED"].str.split().str[0]

    # CPU processing methods
    def process_cpu(self):
        """Process CPU-related features"""
        self.extract_cpu_info()
        self.clean_cpu_threads()
        self.process_cpu_speed()
        self.process_cpu_cores()
        self.process_turbo_clock()
        self.process_clock_multiplier()
        self.process_multithreading()
        self.process_CPU_performance_score()

    def extract_cpu_info(self):
        """Extract CPU information from threads data"""
        cpu_pattern = r"\(([^()]+(?:\([^()]*\))?)\)"
        self.data["CPU"] = (
            self.data["CPU THREADS"]
            .str.extract(cpu_pattern, expand=False)
            .fillna("Undefined")
        )
        self.filter_cpu("Qualcomm Snapdragon")

    def filter_cpu(self, keyword: str):
        """Filter CPUs containing specific keyword"""
        self.data = self.data[~self.data["CPU"].str.contains(keyword, na=False)]

    def clean_cpu_threads(self):
        """Clean CPU threads data"""
        self.data["CPU THREADS"] = self.data["CPU THREADS"].str.split("threads").str[0]

    def process_cpu_speed(self):
        """Process CPU speed data"""
        self.data["CPU SPEED"] = self.data.apply(
            lambda CPU_Detail: (
                re.sub(
                    r"\([^)]*\)",
                    "",
                    find_similar_rows(
                        dataframe=self.data,
                        target_row=CPU_Detail,
                        reference_column="CPU SPEED",
                        matching_columns=["CPU THREADS"],
                        comparison_column="CPU",
                        mode="partial",
                    ),
                ).strip()
                if CPU_Detail["CPU SPEED"] == "Undefined"
                else re.sub(r"\([^)]*\)", "", CPU_Detail["CPU SPEED"]).strip()
            ),
            axis=1,
        )

    def process_cpu_cores(self):
        """Process CPU cores data"""
        # Total CPU Cores - Calculate Total CPU Cores
        self.data["CPU CORES"] = self.data["CPU SPEED"].apply(
            lambda CPU_Detail: self.calculate_total_cpu_cores(CPU_Detail)
        )

    def process_turbo_clock(self):
        """Process turbo clock speed data"""
        # Turbo Clock Speed - Remove "GHz" and CPU's Name
        self.data["TURBO CLOCK SPEED"] = self.data.apply(
            lambda CPU_Detail: (
                find_similar_rows(
                    dataframe=self.data,
                    target_row=CPU_Detail,
                    reference_column="TURBO CLOCK SPEED",
                    matching_columns=["CPU THREADS", "CPU CORES"],
                    comparison_column="CPU",
                    mode="similarity",
                    similarity_threshold=90,
                )
                .split("GHz")[0]
                .strip()
                if CPU_Detail["TURBO CLOCK SPEED"] == "Undefined"
                else CPU_Detail["TURBO CLOCK SPEED"].split("GHz")[0].strip()
            ),
            axis=1,
        )

    def process_clock_multiplier(self):
        """Process clock multiplier data"""
        self.data["GPU CLOCK MULTIPLIER"] = self.data.apply(
            lambda CPU_Detail: (
                find_similar_rows(
                    dataframe=self.data,
                    target_row=CPU_Detail,
                    reference_column="GPU CLOCK MULTIPLIER",
                    matching_columns="CPU",
                    comparison_column="CPU",
                    mode="similarity",
                    similarity_threshold=85,
                )
                .split(" ")[0]
                .strip()
                if CPU_Detail["GPU CLOCK MULTIPLIER"] == "Undefined"
                else CPU_Detail["GPU CLOCK MULTIPLIER"].split(" ")[0].strip()
            ),
            axis=1,
        )

    def process_multithreading(self):
        """Process multithreading data"""
        self.data["USES MULTITHREADING"] = self.data["USES MULTITHREADING"].apply(
            lambda CPU_Detail: 1 if CPU_Detail == "Undefined" else CPU_Detail
        ).replace("No", 0).replace("Yes", 1)

    def process_CPU_performance_score(self):
        """Process performance score data"""
        for Column in ["CPU PASSMARK RESULT (SINGLE)", "CPU PASSMARK RESULT"]:
            self.data[Column] = self.data.apply(
                lambda Score: Score[Column].split(" ")[0], axis=1
            )
        for Column in ["GPU GEEKBENCH 6 RESULT (MULTI)", "GPU GEEKBENCH 6 RESULT (SINGLE)"]:
            self.data[Column] = self.data.apply(
                lambda Score: (
                    find_similar_rows(
                        dataframe=self.data,
                        target_row=Score,
                        reference_column=Column,
                        matching_columns="CPU",
                        comparison_column="CPU",
                        mode="similarity",
                    ).split(" ")[0]
                    if Score[Column] == "Undefined"
                    else Score[Column].split(" ")[0]
                ),
                axis=1,
            )

    def process_gpu(self):
        """Process GPU-related features"""
        self.extract_gpu_info()
        self.gpu_independent_procession()
        self.gpu_dependent_procession()

    def extract_gpu_info(self):
        """Extract GPU name from GPU data"""
        self.data["GPU"] = self.data.apply(self.retrieve_gpu_usage, axis=1)
        self.filter_invalid_gpu()

    def final_processing(self):
        """Final processing steps"""
        self.data.drop(
            self.data[self.data["GPU PASSMARK (G3D) RESULT"] == "Undefined"].index,
            inplace=True,
        )
        self.data.drop(self.data[self.data["BRAND"] == "Casper"].index, inplace=True)

        self.save_data()
    # Helper methods
    def calculate_total_cpu_cores(self, name: str) -> str:
        Total_Cores = 0
        if "&" in name:
            Core_List = name.split("&")
            for Core in Core_List:
                Core_Count = Core.split("x")[0].strip()
                Total_Cores += int(Core_Count)
        elif "x" in name and name != "Undefined":
            Total_Cores = int(name.split("x")[0].strip())
        else:
            Total_Cores = int(name) - 6
        return str(Total_Cores)

    def retrieve_gpu_usage(self, Device_Detail: pd.DataFrame) -> str:
        if (
            isinstance(Device_Detail["DEVICE"], str)
            and len(Device_Detail["DEVICE"].split("/")) == 4
        ):
            return re.sub(
                r"\d+GB",
                "",
                Device_Detail["DEVICE"].split("/")[1].replace("Laptop", "").strip(),
            )
        if "GPU TURBO" in Device_Detail and isinstance(
            Device_Detail["GPU TURBO"], str
        ):
            GPU_Matches = re.findall(
                r"\(([^()]+(?:\([^()]*\))?)\)", Device_Detail["GPU TURBO"]
            )
            if GPU_Matches:
                return re.sub(
                    r"\d+GB", "", GPU_Matches[0].replace("Laptop", "").strip()
                )
        return "Undefined"

    def filter_invalid_gpu(self):
        self.data = self.data[
            (self.data["VRAM OF GPU"] != "Undefined")
            & (self.data["GPU"] != self.data["CPU"])
        ]

    def gpu_independent_procession(self) -> pd.DataFrame:
        for column in self.gpu_struc:
            if column["type"] == "independent":
                col_name = column["name"]
                slitword = column["slitword"]
                self.data[col_name] = (
                    self.data[col_name]
                    .astype(str)
                    .map(lambda data: data.split(slitword)[0].strip())
                )

    def gpu_dependent_procession(self) -> pd.DataFrame:
        for column in self.gpu_struc:
            if column["type"] == "dependent":
                col_name = column["name"]
                slitword = column["slitword"]
                matching = column["matching"]
                comparison = column["comparison"]
                mode = column["mode"]
                threshold = column["threshold"]
                self.data[col_name] = self.data.apply(
                    lambda data: (
                        find_similar_rows(
                            dataframe=self.data,
                            target_row=data,
                            reference_column=col_name,
                            matching_columns=matching,
                            comparison_column=comparison,
                            mode=mode,
                            similarity_threshold=threshold,
                        )
                        .split(slitword)[0]
                        .strip()
                        if data[col_name] == "Undefined"
                        or data["CPU"] in data[col_name]
                        else data[col_name].split(slitword)[0].strip()
                    ),
                    axis=1,
                )

    def reorder_column(self) -> pd.DataFrame:
        """Reorder dataframe columns"""
        for column in self.modified_columns:
            column_name = column["column"]
            if column["rename"]:
                self.data.rename(columns={column["column"]: column["rename"]}, inplace=True)
                column_name = column["rename"]
            if column["before"] or column["after"]:
                self.data = relocate_column(
                    Information=self.data,
                    Column_Name=column_name,
                    Before=column["before"],
                    Behind=column["after"],
                )

    def save_data(self):
        """Save processed data to CSV"""
        self.data.to_csv(
            os.path.join(self.processed_path, "performance_cleaning.csv"), index=False
        )

if __name__ == "__main__":
    performance_cleaning()
