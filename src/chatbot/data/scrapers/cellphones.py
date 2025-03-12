import os, sys
from pathlib import Path

sys.path.append(str(Path().resolve()))
import requests
import time, re
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from seleniumbase import Driver
import pandas as pd
from tqdm.auto import tqdm
from chatbot import pathtree

os.makedirs(
    os.path.join(pathtree("chatbot").get("images"), "cellphones"), exist_ok=True
)

brands = ["asus", "acer", "dell", "lenovo", "hp", "msi", "lg", "gigabyte", "samsung"]
for brand in brands:
    os.makedirs(
        os.path.join(pathtree("chatbot").get("cellphones"), brand),
        exist_ok=True,
    )


class crawl_image:
    browser = "chrome"
    window_size = "1280,720"
    uc = True
    undetectable = True
    incognito = True
    headless = True
    disable_gpu = True
    ad_block = True
    close_popup_flag = False
    close_second_popup_flag = False
    img_save_dir = pathtree("chatbot").get("cellphones")
    url_save_dir = os.path.join(pathtree("chatbot").get("raw"), "img_url.csv")
    URL = "https://cellphones.com.vn/laptop.html?order=filter_price&dir=desc&manufacturer=hp,msi,asus,samsung,acer,dell,lenovo,gigabyte,lg"

    def __init__(self, url: str = None, get_url: bool = False, get_img: bool = False):
        self.browser = Driver(
            browser=self.browser,
            window_size=self.window_size,
            headless=self.headless,
            incognito=self.incognito,
            disable_gpu=self.disable_gpu,
            ad_block=self.ad_block,
            uc=self.uc,
            undetected=self.undetectable,
        )
        self.url = url if url else self.URL

        if get_url:
            self.get_url(url=self.url)
        if get_img:
            self.get_img(url=self.url)
        self.browser.quit()

    def close_popup(self):
        if not self.close_popup_flag:
            try:
                close_promo_button = WebDriverWait(self.browser, 1).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "button.cancel-button-top")
                    )
                )
                close_promo_button.click()
                time.sleep(0.5)
                self.close_popup_flag = True
            except Exception:
                pass
            
        if not self.close_second_popup_flag:
            try:
                close_promo_button = WebDriverWait(self.browser, 1).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "button.modal-close.is-large")
                    )
                )
                close_promo_button.click()
                time.sleep(0.5)
                self.close_second_popup_flag = True
            except Exception:
                pass

    def total_page(self):
        num_product_per_page = 20
        src = BeautifulSoup(self.browser.page_source, "lxml")
        total_num_page = src.find(
            "a", class_="button btn-show-more button__show-more-product"
        )
        total_step = re.search(r"\d+", total_num_page.get_text(strip=True))
        if total_step:
            total_step = int(int(total_step.group()) / num_product_per_page)
        return total_step + 1

    def get_url(self, url: str):
        self.browser.get(url)
        with tqdm(total=2, desc="Closing popup") as main_bar:
            updated_popup = False
            updated_second_popup = False            
            while not self.close_popup_flag or not self.close_second_popup_flag:
                self.close_popup()
                time.sleep(1)
                if self.close_popup_flag and not updated_popup:
                    main_bar.update(1)
                    updated_popup = True
                if self.close_second_popup_flag and not updated_second_popup:
                    main_bar.update(1)
                    updated_second_popup = True
                
        time.sleep(1)
        total_page = self.total_page()
        with tqdm(total=total_page, desc="Crawling product URL") as main_bar:
            for _ in range(total_page):
                try:
                    show_more_button = WebDriverWait(self.browser, 1.5).until(
                        EC.element_to_be_clickable(
                            (
                                By.CSS_SELECTOR,
                                "a.button.btn-show-more.button__show-more-product",
                            )
                        )
                    )
                    show_more_button.click()
                    time.sleep(1)
                    main_bar.update(1)
                except Exception:
                    print("No more button to click")
                    break

        page = self.browser.page_source
        src = BeautifulSoup(page, "lxml")
        all_product = src.find_all("div", class_="product-info-container product-item")
        products = []

        for product in all_product:
            link_div = product.find("div", class_="product-info")
            name_div = product.find("div", class_="product__name")
            name, link = "", ""
            if name_div and link_div:
                name = name_div.find("h3").get_text(strip=True)
                link = link_div.find("a")["href"]
                if (
                    "- Chỉ có tại CellphoneS" in name
                    or "- Nhập khẩu chính hãng" in name
                ):
                    name.replace("- Chỉ có tại CellphoneS", "").replace(
                        "- Nhập khẩu chính hãng", ""
                    )
            products.append({"name": name, "link": link, "get image": False})

        df = pd.DataFrame(products)
        df.to_csv(self.url_save_dir, index=False)
        return df

    def Bypass_Cloudflare(self, url):
        """Bypass Cloudflare"""
        try:
            self.browser.uc_open_with_reconnect(url)
        except Exception:
            self.browser.quit()

    def decode_image(self, img_url: str, save_dir: str):
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            with open(save_dir, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

    def index_brand(self, name: str):
        brands = [
            "asus",
            "acer",
            "dell",
            "lenovo",
            "hp",
            "msi",
            "lg",
            "gigabyte",
            "samsung",
        ]
        index_brand = None
        for brand in brands:
            if brand in name.lower():
                index_brand = brand
                break
        return index_brand

    def get_img(self, url: str = None):
        if not os.path.exists(self.url_save_dir):
            url_lst = self.get_url(url)
        else:
            url_lst = pd.read_csv(self.url_save_dir)
        total_step = len(url_lst[url_lst["get image"] == False])
        with tqdm(total=total_step, desc="Crawling image") as main_bar:
            for _, index in url_lst.iterrows():
                base_name = index["name"]
                if (
                    not index["get image"]
                    and "Zenbook DUO".lower() not in base_name.lower()
                ):
                    main_bar.set_postfix({'Process': base_name})
                    time.sleep(1)
                    img_url_list = []
                    self.browser.get(index["link"])
                    page = self.browser.page_source
                    src = BeautifulSoup(page, "lxml")
                    all_img_url = src.find_all(
                        "div",
                        class_="gallery-slide gallery-top swiper-container swiper-container-initialized swiper-container-horizontal",
                    )
                    for images in all_img_url:
                        img_list = images.find_all("div", class_="swiper-slide")
                        for img in img_list:
                            tag_filter = img.find("a")
                            if (
                                tag_filter
                                and tag_filter.has_attr("href")
                                and "text" in tag_filter["href"]
                            ):
                                img_url_list.append(tag_filter["href"])

                    index_brand = self.index_brand(base_name)
                    for num, img_url in enumerate(img_url_list):
                        name_file = f"{base_name} {num}.jpg"
                        save_dir = os.path.join(
                            self.img_save_dir, index_brand, name_file
                        )
                        self.decode_image(img_url=img_url, save_dir=save_dir)

                    url_lst.at[_, "get image"] = True
                    url_lst.to_csv(self.url_save_dir, index=False)
                    main_bar.update(1)


if __name__ == "__main__":
    _ = crawl_image(get_url=True, get_img=True)
