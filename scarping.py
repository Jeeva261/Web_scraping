import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd

url="https://weather.com/en-IN/weather/tenday/l/Salem+Tamil+Nadu?canonicalCityId=6bac361845d6b7a37c1b5cdc646fe6d09fc9e19fbbb034945b1307421f76106d"

response=requests.get(url)

soup=BeautifulSoup(response.content)

title=soup.find("title").text
print(title)

paragraphs=soup.find("p").text
print(paragraphs)

paragraphs=[p.text for p in soup.find_all("p")]
print(paragraphs,"\n\n\n")
for i,paragraph in enumerate(paragraphs):
    print(f'paragraph{i+1}:{paragraph}')


spans=soup.find_all("span")
for span in spans:
    print(span.prettify(),"\n")



date=[date.text for date in soup.find_all("h2", class_="DailyContent--daypartName--3emSU")]
print(date,"\n\n")



spans=[span.text for span in soup.find_all("span")]  #class_="DailyContent--degreeSymbol--EbEpi")]

for i,snap in enumerate(spans):
    print(f"span{i+1}:{snap}")



span=[span.text for span in soup.find_all("span", dir="ltr")]
print(span,"\n")


# web scraping and taking with dataframe

date_array=pd.DataFrame(date)
print(date_array)
span_array=pd.DataFrame(span)
print(span_array)


data={
    "date":date_array[0],
    # "day":[date],
    "degree":span_array[0]
         
}

df=pd.DataFrame(data)
df.dropna(inplace=True)
print(df)