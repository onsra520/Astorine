import os, time, json5, requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

base_dir = Path(__file__).resolve().parents[1]
storage_dir = os.path.join(base_dir, "storage")
config_dir = os.path.join(base_dir, "config")

paths = {
    "raw": os.path.join(storage_dir, "raw"),
    "images": os.path.join(storage_dir, "images"),
    "config": os.path.join(config_dir, "config.json"),
}

os.makedirs(paths["raw"], exist_ok=True)
os.makedirs(paths["images"], exist_ok=True)

class VersusScraper:
    def __init__(self):
        self.Chrome_Browser = webdriver.Chrome(options=self.LoadOptions())

    def LoadOptions(self):
        """
        Loads the options for the Chrome browser.
        """
        Options = webdriver.ChromeOptions()
        with open(paths["config"], "r", encoding="utf-8") as f:
            Config = json5.load(f)
        arguments = Config["chrome_options"].get("arguments", {})
        for Key, Value in arguments.items():
            if Value == True:
                Options.add_argument(Key)

        Experimental_Options = Config["chrome_options"].get("experimental_options", {})
        for Key, Value in Experimental_Options.items():
            Options.add_experimental_option(Key, Value)

        return Options

    def CollectDeviceLinks(self):
        """
        Collects the URLs of all devices on Versus.com and saves them to a CSV file.

        Returns:
            file: CSV file containing the device names and URLs.

        Note:
            DEVICE: name of the device.
            URL: URL of the device's page.
            STATUS: status of the device (0: not processed, 1: processed).
        """
        Device_Names = []
        Device_URLs = []
        for Page_Number in range(1, 41):
            URL = f"https://versus.com/en/laptop?page={Page_Number}&sort=versusScore"
            Page = requests.get(URL).text
            Source_Code = BeautifulSoup(Page, "lxml")
            Device = Source_Code.find_all("div", class_="Item__item___u6w4M")
            Device_Names += [
                item.find("p", class_="Item__name___QfnBy").text
                for item in Device
                if item.find("p", class_="Item__name___QfnBy")
            ]
            Device_URLs += [
                "https://versus.com"
                + item.find("a", class_="Item__link___GwLFe")["href"]
                for item in Device
                if item.find("a", class_="Item__link___GwLFe")
            ]

        Device_Names = [name.replace('"', " Inches") for name in Device_Names]
        Device_And_URLs = pd.DataFrame(
            {"DEVICE": Device_Names, "VERSUS": Device_URLs, "STATUS": 0}
        )
        Device_And_URLs.to_csv(
            os.path.join(paths["raw"], "Device URLs.csv"),
            mode="w",
            header=True,
            index=False,
            encoding="utf-8",
        )
        print("Collected device URLs.")

    def ClickShowMoreButton(self):
        """Clicks the 'Show More' button on a Versus.com page to load more properties.

        Args:
            Edge_Browser (webdriver): Selenium Edge browser object.
        """
        while True:
            try:
                Wait = WebDriverWait(self.Chrome_Browser, 3)
                Show_More_Button = Wait.until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "div.Group__buttonContainer___hKHBX button")
                    )
                )
                Show_More_Button.click()
            except Exception:
                break

    def ExtractGroupIds(self, Source_Code=None):
        """
        Extracts the group IDs from a Versus.com page.

        Args:
            Source_Code (BeautifulSoup): BeautifulSoup object, source code of the page.

        Returns:
            list: List of group IDs.
        """
        All_Group = []
        for Tag in Source_Code.find_all("div", class_="Group__group___e_mZg"):
            if Tag.has_attr("id"):
                All_Group.append(Tag["id"])
        return All_Group

    def RetrieveGroupInfo(self, Source_Code=None, Group_Name=None, Label_Tables=None):
        """
        Extracts information from a group of properties on a Versus.com page.

        Args:
            Source_Code (BeautifulSoup): BeautifulSoup object, source code of the page.
            Group_Name (str): ID of the group of properties.
            Label_Tables (list): list of property labels to extract.

        Returns:
            dict: Information extracted from the group of properties.
        """

        Extract_Name = Source_Code.find(
            "div", style="transform:translateX(-0%%)", class_="summaryName selected"
        ).text.replace('"', " Inches")
        Information = {"DEVICE": Extract_Name}
        Tags = Source_Code.find("div", id=Group_Name, class_="Group__group___e_mZg")
        if Label_Tables is None:
            Label_Tables = [
                Property.find("span", class_="Property__label___zWFei")
                .text.strip()
                .upper()
                for Property in Tags.find_all(
                    "div", class_="Property__property___pNjSI"
                )
                if Property.find("span", class_="Property__label___zWFei")
            ]
        Label_Tables = [label.upper() for label in Label_Tables]
        for Property in Tags.find_all("div", class_="Property__valueContainer___NYVc0"):
            Label = Property.find("span", class_="Property__label___zWFei")
            if Label and Label.text.strip().upper() in Label_Tables:
                Detail = next(
                    (
                        Property.find(HTML_Tag, class_=class_name)
                        for HTML_Tag, class_name in [
                            ["p", "Number__number___G9V3S"],
                            ["span", "Boolean__boolean_yes___SBedx"],
                            ["span", "Boolean__boolean_no___NI4kH"],
                        ]
                        if Property.find(HTML_Tag, class_=class_name)
                    ),
                    None,
                )
                if Detail and Detail.text:
                    Text = (
                        Detail.text.strip()
                        .replace("✔", "Yes")
                        .replace("✖", "No")
                        .replace("Unknown. Help us by suggesting a value.", "Undefined")
                    )
                    Information[Label.text.strip().upper()] = Text

        return Information

    def StoreDeviceData(self, Source_Code=None):
        """
        Extracts information from a Versus.com page and saves it to a CSV file.

        Args:
            Source_Code (BeautifulSoup): BeautifulSoup object, source code of the page.
        """
        for Group_Tags in self.ExtractGroupIds(Source_Code):
            Name_File = f"Device {Group_Tags.split('_')[1].title()}.csv"
            File_Path = os.path.join(paths["raw"], Name_File)
            New_Columns = list(
                self.RetrieveGroupInfo(
                    Source_Code, Group_Tags, Label_Tables=None
                ).keys()
            )
            if os.path.exists(File_Path):
                Existing_Data = pd.read_csv(File_Path, encoding="utf-8")
                for Col_Name in New_Columns:
                    if Col_Name not in Existing_Data.columns:
                        Existing_Data[Col_Name] = None
                New_Device_Information = pd.DataFrame(
                    [
                        self.RetrieveGroupInfo(
                            Source_Code, Group_Tags, Label_Tables=None
                        )
                    ],
                    columns=Existing_Data.columns,
                )
                Existing_Data = pd.concat(
                    [Existing_Data, New_Device_Information], ignore_index=True
                )
                Existing_Data.to_csv(
                    File_Path, mode="w", header=True, index=False, encoding="utf-8"
                )
            else:
                New_Device_Information = pd.DataFrame(
                    [
                        self.RetrieveGroupInfo(
                            Source_Code, Group_Tags, Label_Tables=None
                        )
                    ],
                    columns=New_Columns,
                )
                New_Device_Information.to_csv(
                    File_Path, mode="w", header=True, index=False, encoding="utf-8"
                )

    def SaveDeviceImage(self, Device_Name=None, Source_Code=None):
        """Saves the image of a device from its Versus.com page.

        Args:
            Device_Name (str): name of the device.
            Source_Code (BeautifulSoup): BeautifulSoup object, source code of the device's page
        """
        try:
            All_Image_URL = Source_Code.find("div", class_=["modernImage"])
            Image_URL = All_Image_URL.find("img")["src"]
            if not Image_URL:
                print(f"Can't find image for {Device_Name}.")
                return

            File_Name = "-".join(Device_Name.replace('"', "").split("/"))
            Image_Name = os.path.basename(File_Name)
            Output_Path = os.path.join(paths["images"], Image_Name)

            response = requests.get(Image_URL, stream=True)
            if response.status_code == 200:
                with open(Output_Path, "wb") as file:
                    for Chunk in response.iter_content(1024):
                        file.write(Chunk)
        except Exception as e:
            print(f"Error saving image for {Device_Name}: {e}")

    def FetchDeviceData(
        self,
    ):
        """
        Fetches data from Versus.com and saves it to CSV files.
        """
        if "Device URLs.csv" not in os.listdir(paths["raw"]):
            print("Device URLs not found. Collecting device URLs.")
            self.CollectDeviceLinks()

        Markup = pd.read_csv(os.path.join(paths["raw"], "Device URLs.csv"))
        try:
            for _, Row in Markup.iterrows():
                if Row["STATUS"] == 0:
                    try:
                        self.Chrome_Browser.get(Row["VERSUS"])
                        print(f"Processing {Row['DEVICE']}...")
                        time.sleep(5)
                        self.ClickShowMoreButton()
                        DevicePageSource = BeautifulSoup(
                            self.Chrome_Browser.page_source, "lxml"
                        )
                        self.StoreDeviceData(DevicePageSource)
                        self.SaveDeviceImage(Row["DEVICE"], DevicePageSource)

                        Markup.at[_, "STATUS"] = 1
                        Markup.to_csv(
                            os.path.join(paths["raw"], "Device URLs.csv"),
                            mode="w",
                            header=True,
                            index=False,
                            encoding="utf-8",
                        )

                    except Exception:
                        print(f"Error processing: {Row['DEVICE']}.")
        except KeyboardInterrupt:
            print("Process interrupted.")
        finally:
            self.Chrome_Browser.quit()
            print("Finished processing all devices.")

if __name__ == "__main__":
    Run = VersusScraper()
    Run.FetchDeviceData()
