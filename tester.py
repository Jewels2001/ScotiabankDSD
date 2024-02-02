import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import Sentiment_Analysis as senti
import Base_Chatbot as classi

data = pd.read_excel("Data/Winter 2024 Scotia DSD Data Set.xlsx")
words = {'Review': [], 'Review_ID': []}

labels = ['well maintained hotel', 'vacation spot', 'lot of food', 'breakfast', 'comfortable room', 'destination', 'nights', 'stay a long time', 'beautiful sights', 'loved the view', 'homie', "good room service", "app", "tech"]

for index, review in enumerate(data['Review']):
    if len(review.split()) > 1:
        labels, scores = classi.zero_shot_classification(review, labels)
        if scores[scores.argmax()] > 0.3 and labels[scores.argmax()] not in ["app", "tech"]:
            words['Review'].append(review)
            words['Review_ID'].append(data['Review_ID'].iloc[index])
            print(review, data['Review_ID'].iloc[index], scores[scores.argmax()] )

output_file_path = 'Easter_egg.xlsx'
df = pd.DataFrame(words)
df.to_excel(output_file_path, index=False)
print(f"Results saved to {output_file_path}")