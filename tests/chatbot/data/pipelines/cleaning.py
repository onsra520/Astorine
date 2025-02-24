import os, json5, time, itertools, re, threading
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from supporter import relocate_column, manage_column_list, find_similar_rows

pd.set_option("future.no_silent_downcasting", True)
base_dir = Path(__file__).resolve().parents[1]

storage_dir = os.path.join(base_dir, "storage")
config_dir = os.path.join(base_dir, "config")

paths = {
    "raw": os.path.join(storage_dir, "raw"),
    "processed": os.path.join(storage_dir, "processed"),
    "config": os.path.join(config_dir, "config.json"),
}

def RemoveMissing():
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

class DatasetFilter:
    def __init__(self, config_path=paths["config"], raw_data_path=paths["raw"]):
        self.config = self._load_config(config_path)
        self.raw_data_path = raw_data_path
        self._load_datasets()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and return configuration from JSON file"""
        with open(config_path, "r") as f:
            return json5.load(f)

    def _load_datasets(self) -> None:
        """Load all datasets specified in the config file"""
        data_config = self.config["data_files"]
        for attribute_name, file_info in data_config.items():
            file_path = os.path.join(self.raw_data_path, file_info["file"])
            df = pd.read_csv(file_path)
            setattr(self, attribute_name, df)

    def __str__(self) -> str:
        return f"DeviceDataProcessor with {len(self.__dict__)-2} loaded datasets"

    def __repr__(self) -> str:
        return self.__str__()

    def _add_column_prefix(
        self, df: pd.DataFrame, prefix: str, exclude: str
    ) -> pd.DataFrame:
        """Add prefix to dataframe columns excluding specified column"""
        return df.rename(
            columns={col: f"{prefix} {col}" for col in df.columns if col != exclude}
        )

    def _filter_na_columns(self, df: pd.DataFrame, max_na_ratio: float) -> pd.DataFrame:
        """Remove columns with NA ratio exceeding threshold"""
        na_ratios = df.isna().mean()
        valid_columns = na_ratios[na_ratios <= max_na_ratio].index.tolist()
        return df[valid_columns]

    def process_performance_data(self) -> pd.DataFrame:
        """Process and merge all performance-related datasets"""
        cpu_df = self._add_column_prefix(self.CPU_Performance, "CPU", exclude="DEVICE")

        # Process GPU data
        gpu_columns = self.config["data_files"]["GPU_Performance"]["columns"]
        gpu_df = self._add_column_prefix(
            self.GPU_Performance[gpu_columns], "GPU", exclude="DEVICE"
        )

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

    def process_device_information(self) -> pd.DataFrame:
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

class PerformanceDataCleaning:
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
        """Initialize and prepare base datasets"""
        performance_data = DatasetFilter().process_performance_data()
        self.data = performance_data.drop(columns=self.get_clean_columns(), errors="ignore")
        self.process_data()

    def get_clean_columns(self) -> List[str]:
        """Manage column configuration in JSON file"""
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

class DesignDataCleaning:
    unnecessary_columns = [
        "HAS A BACKLIT KEYBOARD",
        "USB PORTS",
        "THUNDERBOLT PORTS",
        "HDMI VERSION",
        "DISPLAYPORT OUTPUTS",
        "DOWNLOAD SPEED",
        "UPLOAD SPEED",
        "MINI DISPLAYPORT OUTPUTS",
        "ETHERNET PORTS",
        "HAS AN HDMI OUTPUT",
        "HAS USB TYPE-C",
        "SUPPORTS WI-FI",
        "HAS AN EXTERNAL MEMORY SLOT",
        "BLUETOOTH VERSION",
        "HAS A VGA CONNECTOR",
        "USB 3.0 PORTS",
        "USB 2.0 PORTS",
        "USB 4 20GBPS PORTS",
        "THUNDERBOLT 3 PORTS",
        "USB 3.2 GEN 1 PORTS (USB-C)",
    ]

    reversed_columns = []

    def __init__(self):
        self.config_path = paths["config"]
        self.processed_path = paths["processed"]
        self.initialize_data()

    def initialize_data(self):
        """Initialize and prepare base datasets"""
        design_data = DatasetFilter().process_device_information()
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

class cleaning:
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
            PerformanceDataCleaning()
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
            DesignDataCleaning()
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
            self.reorder_column,
        ]
        for step in processing_steps:
            step()
        self.save_data()

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
            lambda Price: np.floor(self.Generate_Price(Price) * 16681 / 1000) * 1000, axis=1
        )

    def process_cpuspeed(self) -> pd.DataFrame:
        self.data.drop("CPU SPEED", axis=1, inplace=True)
        self.data["CPU SPEED"] = self.data["CPU CLOCK MULTIPLIER"].apply(
            lambda x: x * 100 / 1000
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
            including its brand, CPU, GPU, RAM, and screen size.

        Returns
        -------
        float
            The generated price of the laptop.
        """
        Base_Price = 500

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
        RAM_Price = Product["RAM"] / 16 * 50
        Screen_Size_Price = int(Product["SCREEN SIZE"] - 13) * 50
        Base_Price += CPU_Price + GPU_Price + RAM_Price + Screen_Size_Price
        if Product["BRAND"] in ["Asus", "Dell", "LG", "Microsoft", "Razer", "Samsung"]:
            Base_Price *= 1.2
        elif Product["BRAND"] in ["Acer", "Gigabyte", "HP", "Lenovo", "MSI"]:
            Base_Price *= 1.1
        noise = np.random.uniform(-100, 100)
        final_price = Base_Price + noise
        return round(final_price, 2)

    def save_data(self):
        """Save the merged and cleaned data to a CSV file"""
        self.data.to_csv(
            os.path.join(self.processed_path, "final_cleaning.csv"), index=False
        )


if __name__ == "__main__":
    RemoveMissing()
    cleaning()
