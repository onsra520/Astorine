import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, json5, re
import pandas as pd
from rapidfuzz import fuzz
from fuzzywuzzy import process
from typing import Dict, List
from decimal import Decimal, InvalidOperation
from nlp.helper.ncomp import rlst, srlst, clst, glst, rrlst, dtlst, sslst

project_root = Path(__file__).resolve().parents[1]

paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "odata": os.path.abspath(f"{project_root}/data/storage/processed/final_cleaning.csv"),
    "config" : os.path.abspath(f"{project_root}/config/thresholds.json"),
    "models": os.path.abspath(f"{project_root}/models"),
}

odata = pd.read_csv(paths["odata"])

class ComponentExtractor:
    def __init__(
        self,
        thresholds: Dict[str, float]=None,
        thresholds_name: str="default",
        reset_thresholds: bool=False,
        delete_threshold: str=None,
        ) -> None:
        """
        Initialize a ComponentExtractor instance.

        Parameters:
        thresholds (dict[str, float], optional): A dictionary where the keys are
            component names and the values are the corresponding thresholds.
            Defaults to None.
        thresholds_name (str, optional): The name of the thresholds to load from
            the configuration file. Defaults to "default".
        reset_thresholds (bool, optional): Whether to reset the thresholds to the
            default values. Defaults to False.
        delete_threshold (str, optional): The name of the threshold to delete from
            the configuration file. Defaults to None.

        Returns:
        None
        """
        self.odata = pd.read_csv(paths["odata"])
        self.components = self._load_components()       
        self.thresholds = self._load_thresholds(
            thresholds = thresholds, 
            load=thresholds_name,
            reset=reset_thresholds,
            delete=delete_threshold
            )
        
    def _load_components(self) -> dict:
        """
        Load the components from the odata.

        Returns:
        A dictionary containing the components where the keys are the component
        names and the values are lists of strings representing the components.
        """
        components = {
            "brand": [br.lower() for br in self.odata["BRAND"].unique()],
            "gpu": sorted(glst(), key=len, reverse=False),
            "cpu": sorted(clst(), key=len, reverse=False),
            "ram": sorted(rlst(), key=len, reverse=False),
            "resolution": sorted(srlst(), key=len, reverse=True),
            "refresh rate": sorted(rrlst(), key=len, reverse=False),
            "display type": sorted(dtlst(), key=len, reverse=False),
            "screen size": sorted(sslst(), key=len, reverse=False),
        }
        return components
    
    def _load_thresholds(
        self, 
        thresholds: Dict[str, float]=None, 
        load: str="defaul", 
        reset: bool=False, 
        delete: str=None,
        warn: bool=False
        ) -> dict:
        """
        Load the thresholds from the configuration file.

        Parameters:
        thresholds (dict[str, float], optional): A dictionary where the keys are
            component names and the values are the corresponding thresholds.
            Defaults to None.
        load (str, optional): The name of the thresholds to load from the
            configuration file. Defaults to "default".
        reset (bool, optional): Whether to reset the thresholds to the default
            values. Defaults to False.
        delete (str, optional): The name of the threshold to delete from the
            configuration file. Defaults to None.
        warn (bool, optional): Whether to print a warning message after loading
            the thresholds. Defaults to False.

        Returns:
        A dictionary containing the loaded thresholds.
        """
        if delete is not None:
            del thresholds[delete]
            
        if not os.path.exists(paths["config"]) or reset == True:
            thresholds_load = {"default": {comp: 90 for comp in self.components.keys()}}
        else:
            thresholds_load = json5.load(open(paths["config"], "r"))

        if load not in thresholds_load:
            for comp in self.components.keys():
                if comp not in thresholds.keys():
                    thresholds[comp] = 25
            thresholds_load[load] = thresholds
        json5.dump(thresholds_load, open(paths["config"], "w"), indent=4)
        if warn:
            print(f"{load} is loaded successfully.")
        return thresholds_load[load]

    def _fuzzy_match(self, component: str=None, query: str=None) -> str:
        """
        Perform fuzzy matching on the given query string against the given component.

        Parameters:
        component (str): The name of the component to match against.
        query (str): The query string to match.

        Returns:
        str: The best match if the score is above the threshold, otherwise None.
        """
        min_score = self.thresholds.get(component)
        score_list = process.extractOne(query, self.components[component], scorer=fuzz.WRatio)
        if score_list[1] >= min_score:
            return score_list[0]

    def basic_extract(self, query: str) -> dict:
        """
        Perform basic matching on the given query string against the components.

        Parameters:
        query (str): The query string to match.

        Returns:
        dict: A dictionary containing the extracted components where the keys are
            the component names and the values are the corresponding strings if
            matched, otherwise None.
        """
        query = query.lower()
        extracted= {}
        for comps, values in self.components.items():
            for value in values:
                if value in query:
                    extracted.update({comps: value})
            for comp in self.components.keys():
                if comp not in extracted.keys():
                    extracted.update({comp: None})
        return extracted

    def extract(self, query: str) -> dict:
        """
        Perform fuzzy matching on the given query string against the components.

        Parameters:
        query (str): The query string to match.

        Returns:
        dict: A dictionary containing the extracted components where the keys are
            the component names and the values are the corresponding strings if
            matched, otherwise None.
        """
        query = query.lower()
        extracted = self.basic_extract(query)
        for comp, value in extracted.items():
            if value is None:
                value = self._fuzzy_match(query = query, component = comp)
                extracted[comp] = value
        return extracted
    
class PostProcessor:
    def __init__(self) -> None:
        """
        Initialize the PostProcessor by loading the odata CSV file and converting all
        column names to lowercase.

        """
        self.odata = pd.read_csv(paths["odata"])
        self.odata.columns = [col.lower() for col in self.odata.columns]

    def filter_odata(self, column: str, value: str) -> list:
        """
        Filter the odata based on the given column and value.

        Parameters:
        column (str): The column name to filter.
        value (str): The value to filter.

        Returns:
        list: A list of unique values in the filtered column.
        """
        return self.odata[self.odata[column].astype(str).str.contains(value, case=False, na=False)][column].unique().tolist()
    
    def process_gpu(self, detected_gpu: str) -> list:
        """
        Process the detected GPU string and attempt to match it against known GPUs.

        Parameters:
        detected_gpu (str): The GPU string detected from input data.

        Returns:
        list: A list of matching GPU names from the odata, or None if no matches are found.
        """

        if not detected_gpu:
            return None
        first_peek = self.filter_odata("gpu", detected_gpu)
        if first_peek:
            return first_peek
        if "series" in detected_gpu:
            gpu_match = re.search(r"(RTX\s*\d{2}|RX\s*\d{1}|GTX\s*\d{1})", detected_gpu, re.IGNORECASE)
        else:
            gpu_match = re.search(r"(RTX\s*\d{2}.*|RX\s*\d{1}.*|GTX\s*\d{1}.*)", detected_gpu, re.IGNORECASE)
            
        return self.filter_odata("gpu", gpu_match.group(0)) if gpu_match else None

    def process_cpu(self, detected_cpu: str) -> list:
        """
        Process the detected CPU string and attempt to match it against known CPUs.

        Parameters:
        detected_cpu (str): The CPU string detected from input data.

        Returns:
        list: A list of matching CPU names from the odata, or None if no matches are found.
        """
        if not detected_cpu:
            return None
        normalized = detected_cpu.lower().replace("th", "").replace(" gen", "").replace("series", "").strip()
        cpu_match = self.filter_odata("cpu", normalized)
        if not cpu_match:
            parts = normalized.split()
            if len(parts) > 1:
                normalized = " ".join(parts[:-1])
                cpu_match = self.filter_odata("cpu", normalized)
        return cpu_match if cpu_match else None

    def process_ram(self, detected_ram: str) -> list:
        """
        Process the detected RAM string and attempt to match it against known RAM values.

        Parameters:
        detected_ram (str): The RAM string detected from input data.

        Returns:
        list: A list of matching RAM values from the odata, or None if no matches are found.
        """
        if not detected_ram:
            return None
        normalized = re.findall(r'\d+', detected_ram)
        ram_match = self.filter_odata("ram", str(normalized[0]))
        return ram_match if ram_match else None

    def process_resolution(self, detected_res: str) -> str:
        """
        Process the detected resolution string and match it against known resolutions.

        This function removes specific keywords from the detected resolution string and
        uses fuzzy matching to identify the closest resolution from a predefined dictionary
        of resolutions and their aliases. The function returns the canonical resolution if
        a match is found, otherwise returns None.

        Parameters:
        detected_res (str): The resolution string detected from input data.

        Returns:
        str: The canonical resolution string if a match is found, otherwise None.
        """
        if not detected_res:
            return None
        keywords = ["display", "resolution", "display panel", "display resolution", "screen resolution", "monitor resolution"]
        detected_res = detected_res.lower()
        for kw in keywords:
            detected_res = detected_res.replace(kw, "")
        detected_res = detected_res.strip()
        resolution_dict = {
            "3072 x 1920": ["3072 x 1920", "3k", "3072p", "triple hd"],
            "1920 x 1200": ["1920 x 1200", "wuxga", "16 10 hd+", "hd+ 16 10"],
            "2560 x 1600": ["2560 x 1600", "wqxga", "quad extended 16 10", "retina-like"],
            "2560 x 1440": ["2560 x 1440", "qhd", "quad hd", "2k", "wqhd"],
            "1920 x 1080": ["1920 x 1080", "fhd", "full hd", "1080p"],
            "3840 x 2160": ["3840 x 2160", "4k uhd", "4k", "uhd", "ultra hd", "2160p"],
            "2880 x 1800": ["2880 x 1800", "retina 15", "qhd+ 16 10"],
            "3840 x 2400": ["3840 x 2400", "wquxga", "16 10 4k+"],
            "3200 x 2000": ["3200 x 2000", "qhd+", "3k2k", "wqxga 16 10"],
            "2880 x 1620": ["2880 x 1620", "qhd+ 16 9", "16 9 qhd+", "3k2k 16 9"],
            "3456 x 2160": ["3456 x 2160", "retina 16", "16 inch retina", "3.5k"],
            "2400 x 1600": ["2400 x 1600", "qxga+"],
        }
        resolution_match = None
        for canonical, alias in resolution_dict.items():
            if detected_res in alias:
                resolution_match = self.filter_odata("resolution", str(canonical))
                
        return resolution_match if resolution_match else None

    def process_screen_size(self, detected_ss: str) -> float:
        """
        Process the detected screen size string and match it against known screen sizes.

        This function extracts numerical values from the detected screen size string, 
        converts commas to dots if necessary, and attempts to match the extracted size 
        against known screen sizes from the odata. It returns the matched screen size 
        if found, otherwise returns None.

        Parameters:
        detected_ss (str): The screen size string detected from input data.

        Returns:
        float: The matched screen size if found, otherwise None.
        """
        if not detected_ss:
            return None
        size = re.findall(r'\d+[.,]?\d*', detected_ss)[0].replace(",", ".")
        size_match = self.filter_odata("screen size", str(size))
        return size_match if size_match else None

    def process_refresh_rate(self, detected_rr: str) -> str:
        """
        Extracts the numeric refresh rate value from the detected string.

        This function searches for a numeric value within the given refresh rate string.
        If a numeric value is found, it returns the number as a string. If no number is 
        found or the input is None, it returns None.

        Parameters:
        detected_rr (str): The refresh rate string detected from input data.

        Returns:
        str: The numeric refresh rate value as a string if found, otherwise None.
        """
        if not detected_rr:
            return None
        match = re.search(r'(\d+)', detected_rr)
        return match.group(1) if match else None

    def process_display_type(self, display_type: str) -> list:
        """
        Process the detected display type string and match it against known display types.

        This function converts the detected display type string to lower case and
        attempts to match it against known display types from the odata. If a match is
        found, it returns a list containing the matched display type. If no match is
        found or the input is None, it returns None.

        Parameters:
        display_type (str): The display type string detected from input data.

        Returns:
        list: A list containing the matched display type if found, otherwise None.
        """
        if not display_type:
            return None
        display_type = display_type.lower()
        display_type_match = self.filter_odata("display type", display_type)
        return display_type_match if display_type_match else None
    
    def postprocess(self, detected_components: dict) -> dict:
        """
        Postprocesses the detected components to match them against known values.

        This function takes the detected components and applies the following postprocessing steps:
            - Brand: No postprocessing needed.
            - GPU: Extracts the GPU model name from the detected string and matches it against known GPUs.
            - CPU: Extracts the CPU model name from the detected string and matches it against known CPUs.
            - RAM: Extracts the RAM size from the detected string and matches it against known RAM sizes.
            - Resolution: Extracts the resolution from the detected string and matches it against known resolutions.
            - Screen size: Extracts the screen size from the detected string and matches it against known screen sizes.
            - Refresh rate: Extracts the refresh rate from the detected string and matches it against known refresh rates.
            - Display type: Converts the detected display type string to lower case and matches it against known display types.

        Parameters:
        detected_components (dict): A dictionary of detected components from the input data.

        Returns:
        dict: A dictionary of postprocessed components, where each value is a string representing the matched value from the odata.
        """
        output = {}
        output["brand"] = detected_components.get("brand")
        output["gpu"] = self.process_gpu(detected_components.get("gpu"))
        output["cpu"] = self.process_cpu(detected_components.get("cpu"))
        output["ram"] = self.process_ram(detected_components.get("ram"))
        output["resolution"] = self.process_resolution(detected_components.get("resolution"))
        output["screen size"] = self.process_screen_size(detected_components.get("screen size"))
        output["refresh rate"] = self.process_refresh_rate(detected_components.get("refresh rate"))
        output["display type"] = self.process_display_type(detected_components.get("display type"))
        return output

UPPER_LIMIT_KEYWORDS = [
    "less than", "under", "below", "at most", "up to", "no more than",
    "maximum of", "not exceeding", "spent under", "just under", "barely under",
    "capped at", "limited to", "restricted to", "short of", "falling short of",
    "not surpassing", "not above", "only up to", "maxing out at", "not over",
    "no higher than", "not more than", "ceiling of", "bounded by", "restricted by",
    "confined to", "capped by", "just below", "narrowed to", "finishing at",
    "peaking at", "top limit of", "limit of", "maximum limit of", "no greater than",
    "with a cap of", "reaching up to", "finishing under", "falling under",
    "remaining under", "peaking below", "topping out at", "restricted up to",
    "not exceeding the value of", "limited by", "capped off at", "not surpassing the threshold of",
    "under the threshold of", "with a maximum of"
]

LOWER_LIMIT_KEYWORDS = [
    "at least", "more than", "above", "over", "not less than", "minimum of",
    "no less than", "exceeding", "surpassing", "in excess of", "beyond",
    "starting from", "at a minimum", "a minimum of", "as low as", "greater than",
    "upwards of", "not below", "at the very least", "no lower than",
    "exceeding the minimum", "beyond the floor of", "floor of", "minimum limit of",
    "at a floor of", "rising from", "climbing above", "elevated above",
    "above the minimum", "starting at", "initiating at", "surpassing the minimum of",
    "minimum threshold of", "exceeding the base of", "above base", "beyond the base",
    "not under", "ensuring at least", "no smaller than", "with a floor of",
    "with a minimum of", "starting no lower than", "beginning at", "commencing at",
    "ascending from", "exceeding or equal to", "equal to or more than", "rising above",
    "surmounting", "at the minimum threshold of"
]

RANGE_KEYWORDS = [
    "from", "between", "ranging from", "in the range of", "spanning",
    "extending from", "covering", "stretching from", "starting at", "to",
    "through", "within", "ranging between", "from ... up to", "from ... through",
    "between ... and", "from ... to", "spanning from", "going from", "from a minimum of",
    "from a base of", "bridging", "linking", "connecting", "reaching from",
    "covering a range from", "extending between", "from the low end to", "from the bottom to",
    "from the start to", "from the outset to", "from the minimum to", "between the limits of",
    "among", "ranging over", "from the lower end to", "from the bottom up to",
    "spanning between", "from start through", "from beginning to", "encompassing",
    "inclusive from", "covering from", "transitioning from", "ranging from ... until",
    "stretching between", "bridging between", "from one end to", "from side to side",
    "from lower bound to upper bound"
]

EXACTLY_KEYWORDS = [
    "exactly", "precisely", "just", "exact", "just about", "accurately",
    "no more no less", "to the dot", "to a T", "perfectly", "right at",
    "specifically", "unequivocally", "definitively", "precisely equal to",
    "strictly", "precisely the amount of", "spot on", "absolutely", "exactly equal to",
    "no deviation from", "exact sum of", "on the nose", "precisely matching",
    "precisely the figure of", "to an exact figure", "without variation",
    "unambiguously", "explicitly", "to the exact value", "exact value",
    "right on target", "on point", "accurate to the cent", "exactly the number",
    "with precision", "without any excess", "with exactness", "by the book",
    "in exact terms", "without any deviation", "precisely on", "flawlessly",
    "without discrepancy", "exactly as stated", "precisely as measured",
    "to a precise degree", "down to the last detail", "with pinpoint accuracy",
    "exactly matching the required amount"
]

def extract_prices(text: str) -> List[Decimal]:
    """
    Extract prices from a given text.

    This function takes a text as input and tries to extract all prices from it.
    It supports various formats such as:
      - $1234.56
      - 1234.56$
      - 1234.56 dollar
      - 1234.56 usd

    Returns a list of Decimal objects, each representing a price found in the text.
    """
    pattern = re.compile(
        r"""
        (?:\$(?P<number1>[\d,]+(?:\.\d{1,2})?))                             # $1234.56 or $1234
        |                                                               
        (?:(?P<number2>[\d,]+(?:\.\d{1,2})?)\s*(?:\$|dollars?|usd))         # 1234.56$, 1234.56 dollar, 1234.56 usd,...
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    prices = []
    for match in pattern.finditer(text):
        num_str = match.group("number1") or match.group("number2")
        if num_str:
            num_str = num_str.replace(",", "")
            try:
                prices.append(Decimal(num_str))
            except InvalidOperation:
                continue
    return prices

def parse_price_range(text: str) -> Dict[str, Decimal]:
    """
    Parse a price range from a given text.

    This function takes a text as input and tries to extract a price range from it.
    It supports various formats such as:
      - From X to Y
      - Between X and Y
      - From X up to Y
      - X and above
      - Above X
      - More than X
      - Less than X
      - Under X
      - Below X
      - Exactly X
      - Precisely X

    If the text contains a price range, the function returns a dictionary with two keys: "min" and "max".
    The values of these keys are the minimum and maximum prices in the range, respectively, as Decimal objects.
    If the text does not contain a price range, the function returns a dictionary with two keys: "min" and "max".
    The values of these keys are both 0 as Decimal objects.

    Parameters:
    text (str): The input text to parse the price range from.

    Returns:
    dict: A dictionary with two keys: "min" and "max", containing the minimum and maximum prices in the range, respectively, as Decimal objects.
    """
    text = text.lower()
    range_pattern = re.compile(
        r"(?i)(?:from|between)\s+(?P<min>[\d,]+(?:\.\d{1,2})?)\s+(?:to|and|through|up to)\s+(?P<max>[\d,]+(?:\.\d{1,2})?)"
    )
    m_range = range_pattern.search(text)
    if m_range:
        try:
            min_val = Decimal(m_range.group("min").replace(",", ""))
            max_val = Decimal(m_range.group("max").replace(",", ""))
            return {"min": min_val, "max": max_val}
        except InvalidOperation:
            pass
    exactly_pattern = re.compile(
        r"(?i)(?:" + "|".join(EXACTLY_KEYWORDS) + r")\s+(?P<number>[\d,]+(?:\.\d{1,2})?)"
    )
    m_exact = exactly_pattern.search(text)
    if m_exact:
        try:
            value = Decimal(m_exact.group("number").replace(",", ""))
            return {"min": Decimal(0), "max": value}
        except InvalidOperation:
            pass
    upper_pattern = re.compile(
        r"(?i)(?:" + "|".join(UPPER_LIMIT_KEYWORDS) + r")\s+(?P<number>[\d,]+(?:\.\d{1,2})?)"
    )
    m_upper = upper_pattern.search(text)
    if m_upper:
        try:
            value = Decimal(m_upper.group("number").replace(",", ""))
            return {"min": Decimal(0), "max": value}
        except InvalidOperation:
            pass
    lower_pattern = re.compile(
        r"(?i)(?:" + "|".join(LOWER_LIMIT_KEYWORDS) + r")\s+(?P<number>[\d,]+(?:\.\d{1,2})?)"
    )
    m_lower = lower_pattern.search(text)
    if m_lower:
        try:
            value = Decimal(m_lower.group("number").replace(",", ""))
            return {"min": value, "max": Decimal(0)}
        except InvalidOperation:
            pass
    prices = extract_prices(text)
    if len(prices) >= 2:
        return {"min": min(prices), "max": max(prices)}
    elif prices:
        price_val = prices[0]
        return {"min": Decimal(0), "max": price_val}
    else:
        return {"min": Decimal(0), "max": Decimal(0)}

def extract(
    query: str, 
    thresholds: Dict[str, float]=None, 
    thresholds_name: str=None,
    reset_thresholds: bool=False,
    delete_threshold: str=None
    ) -> dict:
    """
    Extracts components from a given query string using specified thresholds.

    This function initializes a ComponentExtractor with the provided thresholds
    and thresholds name, extracts components from the query, and post-processes
    the results. If no thresholds or thresholds name are provided, default values
    are used. Additionally, it parses price ranges from the query and includes
    them in the post-processed result.

    Parameters
    ----------
    query : str
        The input query string from which components are to be extracted.
    thresholds : Dict[str, float], optional
        A dictionary containing threshold values for each component type.
    thresholds_name : str, optional
        The name of the threshold configuration to load.

    Returns
    -------
    dict
        A dictionary containing the extracted components and their values,
        including parsed price range information.
    """
    extractor = ComponentExtractor(
        thresholds_name= thresholds_name, 
        thresholds = thresholds,
        reset_thresholds=reset_thresholds,
        delete_threshold=delete_threshold
        )
    extracted = extractor.extract(query)
    postprocessor = PostProcessor().postprocess(extracted)
    postprocessor.update({"price":parse_price_range(query)})
    return postprocessor

def use_custom_thresholds():
    thresholds = {
        "brand": 95,
        "gpu": 80,
        'cpu': 80,
        "ram": 90,
        "resolution": 90,
        "refresh rate": 90,
        "display type": 90,
        "screen size": 95,
    } 
    thresholds_name = "custom_threshold"
    return thresholds_name, thresholds

def substring_match(cell_value, keyword):
    """Check if a keyword is a substring of a given cell value.

    Parameters
    ----------
    cell_value : str
        The value of the cell to check.
    keyword : str
        The keyword to search for.

    Returns
    -------
    bool
        True if the keyword is a substring of the cell value, False otherwise.
    """
    return keyword.lower() in cell_value.lower()

def apply_filters(
    query: str=None, 
    weights: dict=None, 
    thresholds_name: str=None, 
    thresholds: dict=None,
    reset_thresholds: bool=False,
    delete_threshold: str=None,
    thresholds_notice: bool=False
    ) -> pd.DataFrame:
    """
    Filter products based on given criteria.

    Parameters
    ----------
    criteria : dict
        A dictionary of criteria to filter by. The keys are the column names, and the values
        are either a list of values to filter by, or a dictionary with keys "min" and/or "max"
        for numerical columns. The values in the list or dictionary should be strings.

    weights : dict, optional
        A dictionary of weights to apply to each field in the criteria. The keys are the same
        as the keys in the criteria dictionary, and the values are the weights to apply. If a
        field is not present in this dictionary, it will be assigned a weight of 0.
    
    thresholds_name : str, optional
        The name of the thresholds to load from the configuration file. Defaults to "default".
    
    thresholds : dict, optional
        A dictionary of thresholds to use for filtering. The keys are the column names, and the
        values are the corresponding thresholds. Defaults to None.
    
    reset_thresholds : bool, optional
        Whether to reset the thresholds to the default values. Defaults to False.
    
    delete_threshold : str, optional
        The name of the threshold to delete from the configuration file. Defaults to None.
    
    thresholds_notice : bool, optional
        Whether to print a notice about the thresholds being used. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the filtered products. If no products match the criteria, an
        empty DataFrame is returned.

    Notes
    -----
    The filtering is done in the following order:
    1. Categorical columns (brand, gpu, cpu, resolution, display type): exact match
    2. Numerical columns (ram, screen size, refresh rate): exact match
    3. Price: range match
    If a product does not match the criteria, it is excluded from the result. If a product does
    not match any of the criteria, but has a non-zero weight for one or more of the fields,
    it will be included in the result with the highest weight.

    Examples
    --------
    >>> filter_products({"brand": "Apple", "price": {"min": 1000, "max": 2000}})
    >>> filter_products({"gpu": ["NVIDIA", "AMD"], "ram": 16}, {"gpu": 2, "ram": 1})
    """
    if query is None:
        raise ValueError("No query provided.")
            
    if thresholds_name:
        threshold_list = json5.load(open(paths["config"], "r", encoding="utf-8"))
        if thresholds_name in threshold_list:
            thresholds = threshold_list[thresholds_name]
    else:
        thresholds_name = "default"
            
    if thresholds_notice:
        print(f"Using thresholds: {thresholds_name}")
        
    criteria = extract(
        query = query, 
        thresholds=thresholds, 
        thresholds_name=thresholds_name,
        reset_thresholds=reset_thresholds,
        delete_threshold=delete_threshold
        )
    original_data = pd.read_csv(paths["odata"])
    if weights is None:
        weights = {}
    original_data.columns = original_data.columns.str.lower()
    mask_all = pd.Series(True, index=original_data.index)
    for field, value in criteria.items():
        if value is None:
            continue
        col = field.lower()
        if field in ["brand", "gpu", "cpu", "resolution", "display type"]:
            if isinstance(value, list):
                mask = original_data[col].astype(str).apply(lambda cell: any(substring_match(cell, crit) for crit in value))
            else:
                mask = original_data[col].astype(str).apply(lambda cell: substring_match(cell, str(value)))
            mask_all = mask_all & mask
        elif field in ["ram", "screen size", "refresh rate"]:
            if isinstance(value, list):
                mask = original_data[col].astype(float).isin(value)
            else:
                mask = original_data[col].astype(float) == float(value)
            mask_all = mask_all & mask
        elif field == "price":
            price_min = value.get("min", Decimal("0"))
            price_max = value.get("max", Decimal("0"))
            if price_min == Decimal("0") and price_max == Decimal("0"):
                continue
            mask = pd.Series(True, index=original_data.index)
            if price_min != Decimal("0"):
                mask = mask & (original_data[col].astype(float) >= float(price_min))
            if price_max != Decimal("0"):
                mask = mask & (original_data[col].astype(float) <= float(price_max))
            mask_all = mask_all & mask

    df_all = original_data[mask_all]
    if not df_all.empty:
        return df_all
    weighted_fields = [
        (field, weights.get(field, 0)) 
        for field in criteria 
        if criteria[field] is not None and weights.get(field, 0) > 0
        ]
    weighted_fields.sort(key=lambda x: x[1], reverse=True)
    
    df_weighted = original_data.copy()
    for field, _ in weighted_fields:
        col = field.lower()
        crit = criteria[field]
        if field in ["brand", "gpu", "cpu", "resolution", "display type"]:
            if isinstance(crit, list):
                mask = df_weighted[col].astype(str).apply(lambda cell: any(substring_match(cell, crit_val) for crit_val in crit))
            else:
                mask = df_weighted[col].astype(str).apply(lambda cell: substring_match(cell, str(crit)))
            df_weighted = df_weighted[mask]
        elif field in ["ram", "screen size", "refresh rate"]:
            if isinstance(crit, list):
                mask = df_weighted[col].astype(float).isin(crit)
            else:
                mask = df_weighted[col].astype(float) == float(crit)
            df_weighted = df_weighted[mask]
        elif field == "price":
            price_min = crit.get("min", Decimal("0"))
            price_max = crit.get("max", Decimal("0"))
            if not (price_min == Decimal("0") and price_max == Decimal("0")):
                mask = pd.Series(True, index=df_weighted.index)
                if price_min != Decimal("0"):
                    mask = mask & (df_weighted[col].astype(float) >= float(price_min))
                if price_max != Decimal("0"):
                    mask = mask & (df_weighted[col].astype(float) <= float(price_max))
                df_weighted = df_weighted[mask]
        if df_weighted.empty:
            break
        
    if df_weighted.shape == original_data.shape:
        return None
    
    return df_weighted
