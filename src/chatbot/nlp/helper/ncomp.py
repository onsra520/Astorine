import os
from pathlib import Path
import pandas as pd
import re

project_root = Path(__file__).resolve().parents[2]

paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "odata": os.path.abspath(
        f"{project_root}/data/storage/processed/final_cleaning.csv"
    ),
}

odata = pd.read_csv(paths["odata"])

def blst():
    """
    Return a list of unique brand names from the spec dataframe in lower case.

    Returns:
        list: A list of unique brand names in lower case.
    """
    return [br.lower() for br in odata["BRAND"].unique().tolist()]

def srlst():
    """
    Generate a list of possible names for a laptop's screen resolution.

    The method first creates a dictionary of resolutions with their corresponding names. Then, it creates a list of the keys and values of the dictionary. It also creates a list of strings that describe the screen, such as "display", "resolution", etc. Then, it creates a list of all possible combinations of the two lists. Finally, it returns the list of all possible names in lower case.

    Returns:
        list: A list of all possible names for a laptop's screen resolution in lower case.
    """
    resolution = {
        "3072 x 1920": ["3072 x 1920", "3K", "3072p", "Triple HD"],
        "1920 x 1200": ["1920 x 1200", "WUXGA", "16:10 HD+", "HD+ (16:10)"],
        "2560 x 1600": ["2560 x 1600", "WQXGA", "Quad Extended (16:10)", "Retina-like"],
        "2560 x 1440": ["2560 x 1440", "QHD", "Quad HD", "2K", "WQHD"],
        "1920 x 1080": ["1920 x 1080", "FHD", "Full HD",  "1080p"],
        "3840 x 2160": ["3840 x 2160", "4K UHD", "4K", "UHD", "Ultra HD", "2160p"],
        "2880 x 1800": ["2880 x 1800", "Retina 15", "QHD+ (16:10)"],
        "3840 x 2400": ["3840 x 2400", "WQUXGA", "16:10 4K+"],
        "3200 x 2000": ["3200 x 2000", "QHD+", "3K2K", "WQXGA+ (16:10)"],
        "2880 x 1620": ["2880 x 1620", "QHD+ 16:9", "16:9 QHD+", "3K2K (16:9)"],
        "3456 x 2160": ["3456 x 2160", "Retina 16", "16-inch Retina", "3,5k", "3K5", "3.5K"],
        "2400 x 1600": ["2400 x 1600", "QXGA+"],
    }
    resolution_values = []
    for key in resolution:
        resolution_values.append(key)
        resolution_values.extend(resolution[key])
    name_1 = list(set(resolution_values))

    sreen_description = [
        "display",
        "resolution",
        "display panel",
        "display resolution",
        "screen resolution",
        "monitor resolution",
    ]
    name_2 = []
    for screen in name_1:
        name_2.extend([f"{x} {y}" for x in [screen] for y in sreen_description])
    screen_name = name_1 + name_2
    return [name.lower() for name in screen_name]

def clst():
    """
    Generate a list of possible names for a laptop's CPU model.

    The method first creates a dictionary of CPU models with their corresponding names. Then, it creates a list of the keys and values of the dictionary. It also creates a list of strings that describe the CPU model, such as "processor", "cpu", etc. Then, it creates a list of all possible combinations of the two lists. Finally, it returns the list of all possible names in lower case.

    Returns:
        list: A list of all possible names for a laptop's CPU model in lower case.
    """
    cpu_data = {
        "intel": {
            "core i5": {}, "core i7": {}, "core i9": {},
            "core ultra 7": {}, "core ultra 9": {},

        },
        "amd": {
            "ryzen 5": {}, "ryzen 7": {}, "ryzen 9": {}, "ryzen 5 pro": {},
            "ryzen 7 pro": {}, "ryzen 9 pro": {}, "ryzen ai 5": {},
            "ryzen ai 7": {}, "ryzen ai 9": {},
        },
    }
    intel, amd= [], []
    for cpu in odata["CPU"]:
        if "Intel" in cpu:
            intel.append(cpu)
        elif "AMD" in cpu:
            amd.append(cpu)

    intel = list(set(intel))
    amd = list(set(amd))

    for intel_cpu in intel:
        for brand_modifier in cpu_data["intel"]:
            sku_numeric_digits = "".join(filter(str.isdigit, intel_cpu.split(" ")[-1]))
            suffix = "".join(filter(str.isalpha, intel_cpu.split(" ")[-1]))
            if brand_modifier not in ["core ultra 7", "core ultra 9"] and brand_modifier in intel_cpu.lower():
                if sku_numeric_digits.startswith("1"):
                    generation = f"{sku_numeric_digits[0:2]} series"
                else:
                    generation = f"{sku_numeric_digits[0:1]} series"
                if generation not in cpu_data["intel"][brand_modifier]:
                    cpu_data["intel"][brand_modifier][generation] = {}
                cpu_data["intel"][brand_modifier][generation][intel_cpu] = suffix

            elif brand_modifier in ["core ultra 7", "core ultra 9"] and brand_modifier in intel_cpu.lower():
                generation = f"{sku_numeric_digits} series"
                if generation not in cpu_data["intel"][brand_modifier]:
                    cpu_data["intel"][brand_modifier][generation] = {}
                cpu_data["intel"][brand_modifier][generation][intel_cpu] = suffix

    name_1 = list(cpu_data["intel"].keys())
    intel_generation = []
    for brand_modifier in name_1:
        intel_generation += list(cpu_data["intel"][brand_modifier].keys())
    intel_generation = list(set(intel_generation))

    for gen in intel_generation:
        if len(gen.split(" ")[0]) <= 2:
            text_1 = gen.replace(" series", "th")
            text_2 = f"gen {gen.replace(' series', '')}"
            intel_generation.append(text_1)
            intel_generation.append(text_2)

    name_2 = []
    pattern_3digits = re.compile(r'^\d{3} series$', re.IGNORECASE)
    for mod in name_1:
        if "ultra" in mod.lower():
            allowed_gen = [g for g in intel_generation if pattern_3digits.match(g)]
            name_2.extend([f"{x} {y}" for x in [mod] for y in allowed_gen])
        else:
            allowed_gen = [g for g in intel_generation if not pattern_3digits.match(g)]
            name_2.extend([f"{x} {y}" for x in [mod] for y in allowed_gen])

    name_3 = [f"intel {mod}" for mod in name_1]
    name_4 = [f"intel {mod}" for mod in name_2]
    name_5 = [cpu.lower() for cpu in set(intel)]
    intel_comp_cpu = name_1 + name_2 + name_3 + name_4 + name_5

    for amd_cpu in amd:
        cpu_lower = amd_cpu.lower()
        for brand_modifier in cpu_data["amd"]:
            if brand_modifier in cpu_lower:
                remove_str = f"amd {brand_modifier}"
                sku_numeric_digits_suffix = cpu_lower.replace(remove_str, "").strip()
                break
        sku_numeric_digits = "".join(filter(str.isdigit, sku_numeric_digits_suffix))
        suffix = "".join(filter(str.isalpha, sku_numeric_digits_suffix))
        if len(sku_numeric_digits) == 4:
            generation = f"{sku_numeric_digits[0:1]} series"
        else:
            generation = f"{sku_numeric_digits} series"

        if generation not in cpu_data["amd"][brand_modifier]:
            cpu_data["amd"][brand_modifier][generation] = {}
        cpu_data["amd"][brand_modifier][generation][amd_cpu] = suffix

    name_1 = list(cpu_data["amd"].keys())
    amd_generation = []
    for brand_modifier in name_1:
        amd_generation += list(cpu_data["amd"][brand_modifier].keys())
    amd_generation = list(set(amd_generation))

    to_remove = []
    for gen in amd_generation:
        if len(gen.split(" ")[0]) <= 2:
            text_1 = gen.replace(" series", "th")
            text_2 = f"gen {gen.replace(' series', '')}"
            amd_generation.append(text_1)
            amd_generation.append(text_2)
            to_remove.append(gen)

    for gen in to_remove:
        amd_generation.remove(gen)

    name_2 = []
    pattern_3digits = re.compile(r'^\d{3} series$', re.IGNORECASE)

    for mod in name_1:
        if "ultra" in mod.lower():
            allowed_gen = [g for g in amd_generation if pattern_3digits.match(g)]
            name_2.extend([f"{x} {y}" for x in [mod] for y in allowed_gen])
        else:
            allowed_gen = [g for g in amd_generation if not pattern_3digits.match(g)]
            name_2.extend([f"{x} {y}" for x in [mod] for y in allowed_gen])

    name_3 = [f"amd {mod}" for mod in name_1]
    name_4 = [f"amd {mod}" for mod in name_2]
    name_5 = [cpu.lower() for cpu in set(amd)]
    amd_comp_cpu = name_1 + name_2 + name_3 + name_4 + name_5

    return intel_comp_cpu + amd_comp_cpu

def gname(brand: str = None, name: str = None, p1: str = " ", p2: str = "") -> str:
    """
    Takes a brand name and a component name and returns a string of the component name with 1 or 2 digits.

    Parameters
    ----------
    brand : str, optional
        The brand name of the component, by default None
    name : str, optional
        The name of the component, by default None
    p1 : str, optional
        The first parameter to add to the returned string, by default " "
    p2 : str, optional
        The second parameter to add to the returned string, by default ""

    Returns
    -------
    str
        The modified string of the component name
    """
    if brand == "nvidia" and name:
        match_rtx = re.search(r"(RTX\s*)(\d{2})\d*", name, re.IGNORECASE)
        if match_rtx:
            return f"{name[:match_rtx.start(2)]}{match_rtx.group(2)}{p1}{p2}"
        match_gtx = re.search(r"(GTX\s*)(\d{1})\d*", name, re.IGNORECASE)
        if match_gtx:
            return f"{name[:match_gtx.start(2)]}{match_gtx.group(2)}0{p1}{p2}"
        return name

    elif brand == "amd" and name:
        match_rx = re.search(r"(RX\s*)(\d{1})\d*", name, re.IGNORECASE)
        if match_rx:
            return f"{name[:match_rx.start(2)]}{match_rx.group(2)}{p1}{p2}"

def glst():
    """
    Generate a list of GPU names, with multiple versions of each name
    to increase the chances of matching the user's query.

    Parameters
    ----------
    None

    Returns
    -------
    list
        A list of GPU names, with multiple versions of each name
    """
    nvidia = odata[odata["GPU"].str.contains("nvidia", case=False, na=False)]["GPU"].unique().tolist()
    amd = odata[odata["GPU"].str.contains("amd", case=False, na=False)]["GPU"].unique().tolist()
    gpu_nvidia, gpu_amd = [], []
    for gpu in nvidia:
        gpu = gpu.lower()
        text_1_1 = gpu
        text_1_2 = gname("nvidia", text_1_1, p2="series").strip()
        text_1_3 = f"{gname('nvidia', text_1_1, p1='00 ', p2='series').strip()}"

        text_2_1 = re.sub(r"\s+", " ", gpu.replace("nvidia", "")).strip()
        text_2_2 = gname("nvidia", text_2_1, p2='series').strip()
        text_2_3 = f"{gname('nvidia', text_2_1, p1='00 ', p2='series').strip()}"

        text_3_1 = re.sub(r"\s+", " ", gpu.replace("geforce", "")).strip()
        text_3_2 = gname("nvidia", text_3_1, p2='series').strip()
        text_3_3 = f"{gname('nvidia', text_3_1, p1='00 ', p2='series').strip()}"

        text_4_1 = re.sub(r"\s+", " ", gpu.replace("nvidia geforce", "")).strip()
        text_4_2 = gname("nvidia", text_3_1, p2='series').strip()
        text_4_3 = f"{gname('nvidia', text_3_1, p1='00 ', p2='series').strip()}"

        texts = [
            text_1_1, text_1_2, text_1_3,
            text_2_1, text_2_2, text_2_3,
            text_3_1, text_3_2, text_3_3,
            text_4_1, text_4_2, text_4_3
            ]
        for text in texts:
            if text:
                gpu_nvidia.append(text)
    nvidia_special_list = [
        "nvidia geforce rtx series", "nvidia geforce gtx series",
        "nvidia rtx series", "nvidia gtx series",
        "geforce rtx series", "geforce gtx series",
        "rtx series", "gtx series", "geforce series",

        ]
    gpu_nvidia.extend(nvidia_special_list)
    gpu_nvidia = list(set(gpu_nvidia))
    if "t500" in gpu_nvidia:
        gpu_nvidia.remove("t500")

    for gpu in amd:
        gpu = gpu.lower()
        text_1_1 = gpu
        text_1_2 = gname("amd", text_1_1, p1="000 ", p2="series")

        text_2_1 = re.sub(r"\s+", " ", gpu.replace("amd", "")).strip()
        text_2_2 = gname("amd", text_2_1, p1="000 ", p2="series")

        text_3_1 = re.sub(r"\s+", " ", gpu.replace("amd radeon", "")).strip()
        text_3_2 = gname("amd", text_3_1, p1="000 ", p2="series")

        text_4_1 = re.sub(r"\s+", " ", gpu.replace("amd radeon", "")).strip()
        text_4_2 = gname("amd", text_3_1, p1="000 ", p2="series")

        texts = [text_1_1, text_1_2, text_2_1, text_2_2, text_3_1, text_3_2, text_4_1, text_4_2]
        for text in texts:
            if text:
                gpu_amd.append(text)
    amd_special_list = [
        "amd rx series", "radeon rx series", "rx series"
    ]
    gpu_amd.extend(amd_special_list)
    gpu_amd = list(set(gpu_amd))

    return gpu_nvidia + gpu_amd

def rlst():
    """
    Generates a list of RAM descriptions with various formats.

    This function retrieves unique RAM values from the 'spec' DataFrame and creates variations of the RAM descriptions
    by rearranging the words and stripping the "GB" suffix. It returns a list of all RAM descriptions in lowercase.

    :return: A list of RAM descriptions with certain prefixes removed, all in lowercase.
    """

    ram_lst = []
    for num in odata["RAM"].unique():
        text_1 = f"{num}GB RAM".strip()
        text_2 = f"RAM {num}GB".strip()
        text_3 = f"{num}GB".strip()
        for text in [text_1, text_2, text_3]:
            ram_lst.append(text)
    return [ram.lower() for ram in ram_lst]

def rrlst():
    """
    Generates a list of refresh rate descriptions with various formats.

    This function retrieves unique refresh rates from the 'spec' DataFrame and creates variations of the refresh rate
    descriptions by rearranging the words and stripping the "Hz" suffix. It returns a list of all refresh rate
    descriptions in lowercase.

    :return: A list of refresh rate descriptions with certain prefixes removed, all in lowercase.
    """
    screen_list = odata["REFRESH RATE"].unique()
    name_1 = [f"{screen}hz" for screen in screen_list]
    name_2 = [f"{screen}hz refresh rate" for screen in screen_list]
    rr_name = name_1 + name_2
    return rr_name

def dtlst():
    """
    Generates a list of display type descriptions with various formats.

    This function retrieves unique display type values from the 'spec' DataFrame and processes each value by removing
    specific prefixes such as "-" and "/". It creates variations of the display type descriptions by rearranging the
    words and stripping the prefixes. Finally, it returns a list of all display type descriptions in lowercase.

    :return: A list of display type descriptions with certain prefixes removed, all in lowercase.
    """
    display_type_list = [dt.lower().replace("-", "").replace("/", " ") for dt in odata["DISPLAY TYPE"].unique()]
    return display_type_list

def sslst():
    """
    Generates a list of screen size descriptions with various formats.

    This function retrieves unique screen size values from the 'spec' DataFrame and processes each value by removing
    specific prefixes such as "inches" and "cm". It creates variations of the screen size descriptions by rearranging
    the words and stripping the prefixes. Finally, it returns a list of all screen size descriptions in lowercase.
    """
    screen_type_list_1 = [f"{str(ss)} inch" for ss in odata["SCREEN SIZE"].unique()]
    screen_type_list_2 = [f"{str(ss).replace('.', ',')} inch" for ss in odata["SCREEN SIZE"].unique()]
    return screen_type_list_1 + screen_type_list_2

def resolution_aliases():
    resolution = {
        "3072 x 1920": ["3K", "3072p", "Triple HD"],
        "1920 x 1200": ["WUXGA", "16:10 HD+", "HD+ (16:10)"],
        "2560 x 1600": ["WQXGA", "Quad Extended (16:10)", "Retina-like"],
        "2560 x 1440": ["QHD", "Quad HD", "2K", "WQHD"],
        "1920 x 1080": ["FHD", "Full HD",  "1080p"],
        "3840 x 2160": ["4K UHD", "4K", "UHD", "Ultra HD", "2160p"],
        "2880 x 1800": ["Retina 15", "QHD+ (16:10)"],
        "3840 x 2400": ["WQUXGA", "16:10 4K+"],
        "3200 x 2000": ["QHD+", "3K2K", "WQXGA+ (16:10)"],
        "2880 x 1620": ["QHD+ 16:9", "16:9 QHD+", "3K2K (16:9)"],
        "3456 x 2160": ["Retina 16", "16-inch Retina", "3,5k", "3K5", "3.5K"],
        "2400 x 1600": ["QXGA+"],
    }
    sreen_description = [
        "display",
        "resolution",
        "display panel",
        "display resolution",
        "screen resolution",
        "monitor resolution",
    ]

    for canonical, aliases in resolution.items():
        extended_aliases = [f"{alias} {desc}" for alias in aliases for desc in sreen_description]
        resolution[canonical].extend(extended_aliases)

    return resolution
