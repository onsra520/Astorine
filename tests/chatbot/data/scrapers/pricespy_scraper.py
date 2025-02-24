import os, time
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from seleniumbase import Driver

base_dir = Path(os.getcwd()).resolve().parents[0]

storage_dir = os.path.join(base_dir, "storage")
config_dir = os.path.join(base_dir, "config")

paths = {
    "raw": os.path.join(storage_dir, "raw"),
    "config": os.path.join(config_dir, "config.json"),
}

Cookie_Accepted = False

class PricespyScraper:
    def __init__(self):
        self.Browser = self.SetupChromeDriver()

    def SetupChromeDriver(self):
        """Setup Chrome WebDriver"""
        Browser = Driver(
            browser="chrome",
            window_size="1280,720",
            uc=True,
            incognito=True,
            headless=True,
            disable_gpu=True,
        )
        return Browser

    def CreatePriceUrl(self, Device_Name):
        """Create PriceSpy URL"""
        Search_Query = (
            Device_Name.replace(" ", "%20").replace("/", "%2F").replace("+", "%2B")
        )
        Search_URL = f"https://pricespy.co.uk/search?search={Search_Query}"
        return Search_URL

    def UpdatePriceURLs(self):
        """Update PriceSpy URLs"""
        file_path = os.path.join(paths["raw"], "Device URLs.csv")
        device_info = pd.read_csv(file_path)
        if "PRICESPY" not in device_info.columns:
            device_info["PRICESPY"] = device_info["DEVICE"].apply(
                lambda x: self.CreatePriceUrl(x)
            )
            col = ["DEVICE", "VERSUS", "PRICESPY", "STATUS"]
            device_info = device_info[col]
            device_info.to_csv(file_path, index=False)
        return device_info

    def Bypass_Cloudflare(self, URL):
        """Bypass Cloudflare"""
        try:
            self.Browser.uc_open_with_reconnect(URL)
        except Exception:
            self.Browser.quit()

    def Cookie_Accept(self):
        """Accept Cookies"""
        try:
            Wait = WebDriverWait(self.Browser, 5)
            Accept_Button = Wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[data-test='CookieBannerAcceptButton']")
                )
            )
            Accept_Button.click()
            time.sleep(1)
        except Exception:
            pass

    def SortByPrice(self):
        """Sort by Price"""
        time.sleep(5)
        try:
            time.sleep(1)
            Wait = WebDriverWait(self.Browser, 3)
            Accept_Button = Wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[data-test='DropdownToggle']")
                )
            )
            Accept_Button.click()

            time.sleep(0.5)
            try:
                Sort_By_Price = Wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//label[text()='Price (High - Low)']")
                    )
                )
                Sort_By_Price.click()
                time.sleep(3)
            except Exception:
                pass
        except Exception:
            pass

    def FetchPriceHistory(self):
        """Fetch Price History"""
        try:
            time.sleep(1)
            Wait = WebDriverWait(self.Browser, 5)
            Title_Element = Wait.until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//a[@data-testid='CardClickableArea']//span[@data-testid='ProductName']",
                    )
                )
            )
            Title_Element.click()
            time.sleep(2)
            try:
                Wait = WebDriverWait(self.Browser, 5)
                Price_History_Button = Wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            "//div[@class='StyledTabLink-sc-0-0 eUsXnE']//span[@data-test='TabButton' and @data-test-section='statistics']",
                        )
                    )
                )
                Price_History_Button.click()
                time.sleep(1)
            except Exception:
                pass
        except Exception:
            pass

    def ExtractDevicePricing(self, URL):
        """Extract Device Pricing"""
        try:
            self.Browser.get(URL)
            self.Cookie_Accept()
            time.sleep(1)
            self.SortByPrice()
            time.sleep(3)

            Device_Source_Code = BeautifulSoup(self.Browser.page_source, "lxml")
            All_Device_Price = Device_Source_Code.find_all(
                "span", class_="Text--q06h0j igDZdP"
            )

            Device_Price = []
            for Price in All_Device_Price:
                Price_Text = Price.text.strip().replace("£", "").replace(",", "")
                Device_Price.append(float(Price_Text))

            if len(Device_Price) == 0:
                self.FetchPriceHistory()
                time.sleep(1)
                Update_Source_Code = BeautifulSoup(self.Browser.page_source, "lxml")
                Price_Element = Update_Source_Code.find(
                    "span", class_="Text--q06h0j ftkWDj StyledText--2v1apx jpPCFb"
                )
                if Price_Element:
                    Device_Price.append(
                        float(
                            Price_Element.text.strip().replace("£", "").replace(",", "")
                        )
                    )

            if Device_Price:
                return str(max(Device_Price))
            else:
                return str("No Price Found")

        except Exception:
            return str("Undefined")

    def FetchDevicePrices(self):
        """Fetch Device Prices"""
        File_Path = os.path.join(paths["raw"], "Device Price.csv")
        Browser = self.SetupChromeDriver()
        try:
            pricespy_urls = self.UpdatePriceURLs()
            device_price = pricespy_urls[["DEVICE"]].copy()
            for _, Row in pricespy_urls.iterrows():
                Index = pricespy_urls.index[
                    pricespy_urls["DEVICE"] == Row["DEVICE"]
                ].tolist()[0]
                Price = self.ExtractDevicePricing(Row["PRICESPY"])
                try:
                    device_price.at[Index, "PRICESPY"] = str(Price)
                    print(f"Price for {Row['DEVICE']} is {Price}")
                except ValueError:
                    print(f"Error updating price for {Row['DEVICE']}")
                    device_price.at[Index, "PRICESPY"] = np.nan

                device_price.to_csv(
                    File_Path, mode="w", header=True, index=False, encoding="utf-8"
                )

        except KeyboardInterrupt:
            print("Process stopped.")
            Browser.quit()
        finally:
            Browser.quit()

if __name__ == "__main__":
    run = PricespyScraper()
    run.FetchDevicePrices()