import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.preprocessing  import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

url="https://www.tutorialrepublic.com/html-tutorial/html-paragraphs.php"

reesponse=requests.get(url)

soup=BeautifulSoup(reesponse.content)

title=soup.title.string

paragraphs=[p.text
for p in soup.find_all("p")]



print(f'tittle of website:{title}')

print("\nparagraphs")
for i, paragraph in enumerate(paragraphs):
    print(f'paragraph{i+1}:{paragraph}')



data={
    "paragraph":paragraphs,
    "count words":[len(p.split()) for p in paragraphs]
}

df=pd.DataFrame(data)
print(df)



encoding=LabelEncoder()
df["paragraph_new"]=encoding.fit_transform(df["paragraph"])
df.drop(columns="paragraph",axis=1,inplace=True)
df.replace(0,np.nan,inplace=True)
df.dropna(inplace=True)
print(df)


x=df[["paragraph_new"]]
y=df["count words"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=20)

x_poly=poly.fit_transform(x)

model=LinearRegression()
res=model.fit(x_poly,y)

y_pred_poly=model.predict(x_poly)





model=LinearRegression()
res=model.fit(x_train,y_train)



print(f"slope:{model.coef_}")
print(f"intercept:{model.intercept_}")


y_pred=model.predict(x_test)
print(y_pred)

r2=r2_score(y_test,y_pred)
print(f'r square:{r2}')

mse=mean_squared_error(y_test,y_pred)
print(f'mean square error:{mse}')


plt.scatter(x_train,y_train,label="Paragraph number",color="blue")
plt.plot(x_train,model.predict(x_train),label="count of paragraph",color="red")
plt.title("Paragraph number vs count of paragraph")
plt.xlabel("Paragraph number")
plt.ylabel("count of paragraph")
plt.legend()
plt.show()


plt.scatter(x,y,label="Paragraph number",color="blue")
plt.plot(x,y_pred_poly,label="count of paragraph",color="red")
plt.title("Paragraph number vs count of paragraph")
plt.xlabel("Paragraph number")
plt.ylabel("count of paragraph")
plt.legend()
plt.show()






new_selection=4.0
new_selection_df=model.predict([[new_selection]])
print(new_selection_df[0])


