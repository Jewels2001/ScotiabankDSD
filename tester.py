import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_excel("Data/Winter 2024 Scotia DSD Data Set.xlsx")
print(df.head())
print(df.dtypes)
rating_mean = df['Rating'].mean()
rating_mode = df['Rating'].mode()
rating_median = df['Rating'].median()
rating_std = df['Rating'].std()

print("The mean rating is: ", rating_mean)
print("The most common rating is: ", rating_mode)
print("The median rating is: ", rating_median)
print("The standard deviation of the rating is: ", rating_std)
words = []
for i in df['Review']:
    if len(i.split()) == 1:
        words.append(i)
print(words)