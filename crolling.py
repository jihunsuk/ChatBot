from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# 25page의 52456번 엑시트 스피드부터 다운로드하면됨.

driver = webdriver.Chrome('C:/Users/tjrwl/PycharmProjects/ChatBot/chromedriver')
driver.implicitly_wait(10)

driver.get('http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption&sca=%ED%95%9C%EA%B8%80')
driver.implicitly_wait(10)
time.sleep(1)

page = driver.find_element_by_class_name("pagination").find_elements_by_tag_name("li")
page[12].find_element_by_tag_name('a').click()

page = driver.find_element_by_class_name("pagination").find_elements_by_tag_name("li")
page[12].find_element_by_tag_name('a').click()

page = driver.find_element_by_class_name("pagination").find_elements_by_tag_name("li")
page[6].find_element_by_tag_name('a').click()

WebDriverWait(driver, 30) \
        .until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
list = driver.find_elements_by_class_name('list-subject')
driver.implicitly_wait(10)

subtitle_index = 0
page_index = 7
last_index = list.__len__()

subtitle_number = 0

while True:
    list[subtitle_index].find_element_by_tag_name('a').click()
    driver.implicitly_wait(10)
    time.sleep(0.3)

    while True:
        try:
            driver.find_element_by_class_name('view_file_download').click()
            driver.implicitly_wait(5)
            time.sleep(0.4)
            break
        except:
            driver.back()
            WebDriverWait(driver, 30) \
                .until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
            list = driver.find_elements_by_class_name('list-subject')
            subtitle_index += 1
            list[subtitle_index].find_element_by_tag_name('a').click()
            driver.implicitly_wait(10)
            time.sleep(0.3)

    driver.find_element_by_tag_name('h3').find_element_by_tag_name('a').click()
    driver.implicitly_wait(10)
    time.sleep(0.45)

    subtitle_number += 1
    print("자막 개수 :", subtitle_number)

    driver.back()
    driver.implicitly_wait(10)
    time.sleep(0.3)

    driver.back()
    driver.implicitly_wait(10)
    time.sleep(0.3)

    WebDriverWait(driver, 30) \
        .until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
    list = driver.find_elements_by_class_name('list-subject')
    driver.implicitly_wait(10)
    time.sleep(0.5)

    subtitle_index += 1

    if subtitle_index == last_index:
        subtitle_index = 0
        page = driver.find_element_by_class_name("pagination").find_elements_by_tag_name("li")
        page[page_index].find_element_by_tag_name('a').click()
        driver.implicitly_wait(10)
        time.sleep(0.5)

        WebDriverWait(driver, 30) \
            .until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
        list = driver.find_elements_by_class_name('list-subject')
        last_index = list.__len__()
        page_index += 1
        if page_index == 13:
            page_index = 3