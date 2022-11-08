import datetime
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common import NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement
import sqlite3
import json
from tqdm import tqdm

### VARIABLES ###

# 뉴스 기사를 수집하는 시점입니다.
# dt = datetime.datetime.now().strftime('%Y%m%d')
dt = '20170310'

###

print('loading database')

with open('./datas/media.json', 'r', encoding='UTF-8') as f:
    medias = json.load(f)

conn = sqlite3.connect('./collection/data-{}.db'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS News(headline text, media text, dt text, link text, content text)')

print('setup chromedriver')

chrome_options = webdriver.ChromeOptions()

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

count = 0
success_count = 0
ignore_count = 0

print('start collecting')

for media in tqdm(medias['medias'], desc='collecting medias'):
    print(f'start collecting media `{media["name"]}`')
    page = 1

    while True:
        url = 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid={}&date={}&page={}'.format(str(media['code']).zfill(3), dt, page)
        driver.get(url)

        try:
            current_page_button = driver.find_element(By.CSS_SELECTOR, '#main_content > div.paging > strong')
        except NoSuchElementException:
            print('ignoring page', page, 'in', media)
            break

        if current_page_button.text != str(page):
            break

        try:
            media_name = driver.find_element(By.CSS_SELECTOR, '#main_content > div.list_body.newsflash_body > div.newsflash_header3 > h3').text
        except:
            continue

        try:
            headlines_1 = driver.find_element(By.CSS_SELECTOR, '#main_content > div.list_body.newsflash_body > ul.type06_headline')
            headlines_1_elements = headlines_1.find_elements(By.TAG_NAME, 'li')
        except NoSuchElementException:
            headlines_elements = []

        try:
            headlines_2 = driver.find_element(By.CSS_SELECTOR, '#main_content > div.list_body.newsflash_body > ul.type06')
            headlines_2_elements = headlines_2.find_elements(By.TAG_NAME, 'li')
        except NoSuchElementException:
            headlines_2_elements = []
            

        headlines_elements: List[WebElement] = headlines_1_elements + headlines_2_elements

        for headline_element in headlines_elements:
            count += 1

            try:
                headline = headline_element.find_elements(By.TAG_NAME, 'a')[1]
                txt = headline.text
                link = headline.get_attribute('href')

                cur.execute('INSERT INTO News VALUES (?, ?, ?, ?, ?)', (txt, media_name, dt, link, None))
                success_count += 1

            except:
                ignore_count += 1
        
        page += 1

print(f'collecting: all {count}, success {success_count}, ignoring {ignore_count} rows')
print('deleting duplicated articles...')

cur.execute('DELETE FROM News WHERE rowid NOT IN (SELECT min(rowid) FROM News GROUP BY headline, media, dt)')
deleted_count = cur.rowcount

print(f'deleted {deleted_count} duplicated rows')

print('committing database')

conn.commit()

print('\ncollecting article content...')

content_count = 0
success_content_count = 0
ignore_content_count = 0

news = cur.execute('SELECT * FROM News')
links = list(map(lambda x: x['link'], news))

print('start to collecting', len(links), 'content')

for link in tqdm(links, desc='collecting contents'):
    driver.get(link)

    article_elements = driver.find_elements(By.ID, 'newsct_article') + driver.find_elements(By.ID, 'newsEndContents')

    content_count += 1

    if not article_elements:
        ignore_content_count += 1
        continue
    
    success_content_count += 1

    article = article_elements[0]

    cur.execute('UPDATE News SET content=? WHERE link=?', (article.text.split('기사제공')[0], link))

print(f'content collecting: all {content_count}, success {success_content_count}, ignoring {ignore_content_count} rows')


conn.commit()
conn.close()

input('\n====\n\nProgram ended')