import sys
import os
from pathlib import Path
import pandas as pd
import re
import json5
import random

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from nlp.qbuilder import assembler


paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "qfragments": os.path.abspath(f"{project_root}/intents/qfragments.json"),
    "questions": os.path.abspath(f"{project_root}/intents/questions.csv"),
}

if "qfragments.json" not in os.listdir(f"{project_root}/intents"):
    assembler()

spec = pd.read_csv(os.path.join(paths["processed"], "final_cleaning.csv"))

txt = json5.load(open(paths["qfragments"], "r", encoding="utf-8"))

def intel_cpu_generation():
    """
    Generates a random string describing an Intel processor.

    There are 4 different types of strings that can be generated:

    1. Just the processor model (e.g. "Core i7")
    2. The processor model followed by the processor generation (e.g. "Core i7 10th")
    3. The processor model with "Intel" prefix followed by the processor generation (e.g. "Intel Core i7 10th")
    4. A random processor model from the dataset (e.g. "intel core i9 12900H")

    The function first checks if the processor model is in the "ultra" category,
    and if so, only allows generations that are 3 digits.
    Then it randomly selects one of the above options and returns the string.

    :return: A string describing an Intel processor
    """

    intel_modifier = list(txt["cpu"]["intel"].keys())
    intel_generation = []
    for brand_modifier in intel_modifier:
        intel_generation += list(txt["cpu"]["intel"][brand_modifier].keys())
    intel_generation = list(set(intel_generation))

    for gen in intel_generation:
        if len(gen.split(" ")[0]) <= 2:
            text_1 = gen.replace(" series", "th")
            text_2 = f"gen {gen.replace(' series', '')}"
            intel_generation.append(text_1)
            intel_generation.append(text_2)

    pattern_3digits = re.compile(r'^\d{3} series$', re.IGNORECASE)

    mod = random.choice(intel_modifier)
    if "ultra" in mod.lower():
        allowed_gen = [g for g in intel_generation if pattern_3digits.match(g)]
    else:
        allowed_gen = [g for g in intel_generation if not pattern_3digits.match(g)]
    if allowed_gen:
        gen = random.choice(allowed_gen)
    else:
        gen = ""
    choice = random.choice([1, 2, 3, 4])
    if choice == 1:
        return mod
    elif choice == 2:
        return f"{mod} {gen}".strip()
    elif choice == 3:
        return f"intel {mod} {gen}".strip()
    else:
        return random.choice(
            spec[spec["CPU"].str.contains("Intel", case=False)]["CPU"].unique()
        ).lower()

def amd_cpu_generation():
    """
    This function generates a random string describing an AMD processor.
    There are 4 different types of strings that can be generated:

    1. Just the processor model (e.g. "Ryzen 5")
    2. The processor model followed by the processor generation (e.g. "Ryzen 5 5600X")
    3. The processor model followed by the processor generation with the word "series" replaced by "th" or "gen" (e.g. "Ryzen 5 5000th" or "Ryzen 5 gen 5")
    4. A random processor model from the dataset (e.g. "AMD Ryzen 5 5600H")

    The function first checks if the processor model is in the "ultra" category, and if so, only allows generations that are 3 digits.
    Then it randomly selects one of the above options and returns the string.

    :return: A string describing an AMD processor
    """
    amd = list(txt["cpu"]["amd"].keys())
    amd_generation = []
    for brand_modifier in amd:
        amd_generation += list(txt["cpu"]["amd"][brand_modifier].keys())
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

    pattern_3digits = re.compile(r'^\d{3} series$', re.IGNORECASE)

    mod = random.choice(amd)
    if "ultra" in mod.lower():
        allowed_gen = [g for g in amd_generation if pattern_3digits.match(g)]
    else:
        allowed_gen = [g for g in amd_generation if not pattern_3digits.match(g)]
    if allowed_gen:
        gen = random.choice(allowed_gen)
    else:
        gen = ""
    choice = random.choice([1, 2, 3, 4])
    if choice == 1:
        return mod
    elif choice == 2:
        return (f"{mod} {gen}".strip())
    elif choice == 3:
        return (f"amd {mod} {gen}".strip())
    else:
        return random.choice(
            spec[spec["CPU"].str.contains("AMD", case=False)]["CPU"].unique()
        ).lower()

def cpu_generation():
    """
    Generates a random string describing a CPU.

    This function randomly selects between an Intel or AMD processor description
    by calling the respective generation functions for each brand. It combines
    the results into a list and returns a randomly chosen string from the list.

    :return: A string describing a CPU, either Intel or AMD
    """
    components = [intel_cpu_generation(), amd_cpu_generation()]
    return random.choice(components)

def gpu_generation():
    """
    Generates a random GPU description string.

    This function processes the unique GPU names from the dataset by removing certain prefixes such as "Nvidia GeForce",
    "AMD Radeon", "Nvidia", "AMD", and "GeForce". It creates variations of the GPU names by stripping these prefixes and
    appends them to the list of GPU descriptions. Finally, it returns a randomly selected GPU description from the list.

    :return: A string describing a GPU
    """
    gpu_text = spec["GPU"].unique().tolist()
    for gpu in gpu_text.copy():
        text_1 = re.sub(r"\s+", " ", gpu.replace("Nvidia GeForce", "")).strip()
        text_2 = re.sub(r"\s+", " ", gpu.replace("AMD Radeon", "")).strip()
        text_3 = re.sub(r"\s+", " ", gpu.replace("Nvidia", "")).strip()
        text_4 = re.sub(r"\s+", " ", gpu.replace("AMD", "")).strip()
        text_5 = re.sub(r"\s+", " ", gpu.replace("GeForce", "")).strip()
        texts = [
            text_1,
            text_2,
            text_3,
            text_4,
            text_5,
        ]
        for text in texts:
            if text:
                gpu_text.append(text)
    return random.choice(gpu_text).lower()


def ram_generation():
    """
    Generates a random string describing the RAM of a laptop.

    This function takes the unique RAM values from the dataset and generates
    variations of the RAM descriptions by rearranging the words and stripping
    the "GB" suffix. It returns a randomly chosen RAM description from the
    list.

    :return: A string describing the RAM of a laptop
    """
    ram_lst = []
    for num in spec["RAM"].unique():
        text_1 = f"{num}GB RAM".strip()
        text_2 = f"RAM {num}GB".strip()
        text_3 = f"{num}GB".strip()
        for text in [text_1, text_2, text_3]:
            ram_lst.append(text)
    return random.choice(ram_lst).lower()

def screen_generation():
    """
    Generates a random string describing a screen resolution.

    This function creates variations of screen resolution descriptions by
    combining resolution options from the dataset with various descriptive
    terms such as "display", "resolution", and "monitor resolution". It
    returns a randomly selected description from the generated list of
    screen resolution strings.

    :return: A string describing a screen resolution
    """

    screen_list = list(txt["resolution"].keys())
    sreen_description = [
        "display",
        "resolution",
        "display panel",
        "display resolution",
        "screen resolution",
        "monitor resolution",
    ]
    for screen in list(txt["resolution"].keys()):
        for option in txt["resolution"][screen]:
            text_1 = f"{option} {random.choice(sreen_description)}"
            text_2 = f"{random.choice(sreen_description)} {option}"
            for text in [text_1, text_2]:
                screen_list.append(text)
    random.shuffle(screen_list)
    return random.choice(screen_list).lower()

def rr_generation():
    """
    Generates a random string describing a refresh rate.

    This function selects a unique refresh rate from the dataset and
    returns it as a string with the 'hz' suffix appended.

    :return: A string describing a refresh rate in hertz
    """
    screen_list = spec["REFRESH RATE"].unique()
    return f"{random.choice(screen_list)}hz"


def format_money(unit, amount=None, known=True):
    """
    Returns a formatted money string.
    If known is True, amount must be provided and will be formatted with the unit.
    If known is False, a placeholder “unknown” value is returned.
    """
    if known and amount is not None:
        if unit == "$":
            return f"${amount}"
        elif unit in ["USD", "dollars"]:
            return f"{amount} {unit}"
        else:
            return str(amount)
    else:
        if unit == "$":
            return "$unknown"
        elif unit in ["USD", "dollars"]:
            return f"unknown {unit}"
        else:
            return "unknown"

def price_generation():
    """
    Generates a random sentence related to pricing using predefined templates.

    This function randomly selects a money-related sentence template from the
    "money" key in the txt dictionary. It then randomly determines a currency
    unit and whether the amount should be known or unknown. If the template
    contains a single money placeholder, it replaces it with a formatted
    money string based on the chosen method. If the template contains two
    money placeholders, it determines different methods for replacing each
    placeholder, potentially using known or unknown values. The function
    returns the generated sentence with placeholders filled in.

    :return: A string with money-related placeholders replaced by formatted values.
    """

    money_sentence = random.choice(txt["money"])
    money_unit = random.choice(["USD", "dollars", "$", "unknown"])
    method = random.choice(["unknown", "known", "known"])

    if "[money]" in money_sentence:
        if method == "known":
            replacement = format_money(
                money_unit, random.randint(1300, 3500), known=True
            )
        else:
            replacement = format_money(money_unit, known=False)
        money_sentence = money_sentence.replace("[money]", replacement)
    elif "[money_1]" in money_sentence and "[money_2]" in money_sentence:
        sub_method = random.choice(["value 1", "value 2", "other"])
        if sub_method == "value 1":
            replacement1 = format_money(money_unit, known=False)
            replacement2 = format_money(
                money_unit, random.randint(1300, 3500), known=True
            )
        elif sub_method == "value 2":
            replacement1 = format_money(
                money_unit, random.randint(1300, 3500), known=True
            )
            replacement2 = format_money(money_unit, known=False)
        else:
            if money_unit == "$":
                replacement1 = f"${random.randint(1300, 1900)}"
                replacement2 = f"${random.randint(1900, 3500)}"
            elif money_unit in ["USD", "dollars"]:
                replacement1 = f"{random.randint(1300, 1900)} {money_unit}"
                replacement2 = f"{random.randint(1900, 3500)} {money_unit}"
            else:
                replacement1 = str(random.randint(1300, 3500))
                replacement2 = str(random.randint(1300, 3500))
        money_sentence = money_sentence.replace("[money_1]", replacement1)
        money_sentence = money_sentence.replace("[money_2]", replacement2)

    return money_sentence


def generate_text(num: int = 100, choice: str = "all", save: bool = False, filename: str = "generated.csv") -> pd.DataFrame:
    """
    Generates a given number of random questions based on the templates, sub-questions, and use-cases defined in the labels.json file.

    The function takes an optional argument of the number of questions to generate, defaulting to 100.

    For each question, the function randomly selects a template, a sub-question, and a use-case from the labels.json file. The function then randomly selects a subset of the following components: cpu, gpu, ram, display, and refresh rate. The selected components are then shuffled and combined into a single string.

    The function generates two possible questions for each combination of components: one using the template and one using the sub-question. The function then randomly selects one of the two questions and adds it to the list of generated questions.

    The function returns a pandas DataFrame containing the generated questions. The DataFrame is also written to the questions.csv file in the project's data directory.

    :param num: The number of questions to generate (default: 100)
    :return: A pandas DataFrame containing the generated questions
    """
    generated_questions = []
    for _ in range(num):
        template = random.choice(txt["templates"])
        sub_template = random.choice(txt["sub questions"])
        use_case = random.choice(txt["use case"])
        sub_brand = random.choice(txt["sub brand"])
        brand = random.choice(spec["BRAND"].unique()).lower()
        price = price_generation()

        connectors = [
            "and", ",", ";", "&", "with", "as well as",
            "plus", "together with", "along with", "as well",
            "in addition to", "besides", "not to mention",
            "accompanied by", "coupled with", "combined with",
            "joined by", "alongside", "together alongside", "next to"
        ]

        components = {
            "cpu": cpu_generation(),
            "gpu": gpu_generation(),
            "ram": ram_generation(),
            "display": screen_generation(),
            "refresh rate": rr_generation(),
        }

        valid_components = {
            component: value for component, value in components.items() if value
        }
        if not valid_components:
            continue

        valid_component_keys = list(valid_components.keys())
        random.shuffle(valid_component_keys)
        number_to_select = random.randint(1, len(valid_component_keys))

        selected_components = {
            component: valid_components[component]
            for component in valid_component_keys[:number_to_select]
        }
        unselected_components = {
            component: None
            for component in components
            if component not in selected_components
        }
        annotation = {}
        annotation.update(selected_components)
        annotation.update(unselected_components)

        comps = list(selected_components.keys())
        random.shuffle(comps)

        component_text = [f"{valid_components[comp]}" for comp in valid_components]
        selected_connectors = random.sample(
            connectors * (len(component_text) // len(connectors) + 1),
            len(component_text) - 1,
        )
        component = " ".join(word + " " + connector for word, connector in zip(component_text[:-1], selected_connectors))+ " " + component_text[-1]
        brand_sentence = sub_brand.replace("[brand]", brand)

        question_entry = {}
        if choice == "all":
            sentence_1 = f"{template.replace('[component]', component).replace('[sub_brand]', brand_sentence)} {use_case} {price}."
            sentence_2 = f"{sub_template.replace('[component]', component).replace('[use_case]', use_case).replace('[sub_brand]', brand_sentence)} {price}."
            question_entry = {"question": random.choice([sentence_1, sentence_2])}
        elif choice == "s1":
            sentence_1 = f"{template.replace('[component]', component).replace('[sub_brand]', brand_sentence)} {use_case} {price}."
            question_entry = {"question": sentence_1}
        elif choice == "s2":
            sentence_2 = f"{sub_template.replace('[component]', component).replace('[use_case]', use_case).replace('[sub_brand]', brand_sentence)} {price}."
            question_entry = {"question": sentence_2}
        
        brand_entry = {"brand": brand}
        price_entry = {"price": price}

        entry_list = [valid_components, brand_entry, price_entry]
        for entry in entry_list:
            question_entry.update(entry)

        generated_questions.append(question_entry)

    df = pd.DataFrame(generated_questions)
    
    if save:
        if filename:
            df.to_csv(os.path.join(paths["processed"], filename), index=False, encoding="utf-8")
        else:
            df.to_csv(paths["questions"], index=False, encoding="utf-8")
            
    return df

