import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

data = pd.read_csv("dataset_tiktok-comments-scraper_2026-05-09_08-19-48-508.csv")

print("FIRST 5 ROWS OF DATASET")
print(data.head())

data = data.drop_duplicates()

data = data.dropna()

data['text'] = data['text'].str.lower()

print("\nDATA AFTER CLEANING")
print(data.head())

data['text'] = data['text'].apply(
    lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x))
)

data['text'] = data['text'].str.strip()

print("\nDATA AFTER PREPROCESSING")
print(data['text'].head())

def detect_emotion(text):

    if any(word in text for word in ['worried', 'concern', 'scared', 'job', 'problem']):
        return 'Concern'

    elif any(word in text for word in ['angry', 'mad', 'hate', 'wrong']):
        return 'Anger'

    elif any(word in text for word in ['funny', 'hilarious', 'lol', 'lmao']):
        return 'Amusement'

    elif any(word in text for word in ['happy', 'love', 'great', 'good']):
        return 'Happiness'

    elif any(word in text for word in ['sorry', 'sad', 'poor']):
        return 'Sympathy'

    elif any(word in text for word in ['disappointed', 'bad']):
        return 'Disappointment'

    else:
        return 'Other'

data['emotion'] = data['text'].apply(detect_emotion)

emotion_counts = data['emotion'].value_counts()

print("\nEMOTION DISTRIBUTION")
print(emotion_counts)

data.to_csv("emotion_result.csv", index=False)

print("\nNEW DATASET SAVED SUCCESSFULLY")

emotion_counts.plot(kind='bar')

plt.title("Emotion Distribution from TikTok Comments")
plt.xlabel("Emotion")
plt.ylabel("Frequency")

plt.show()
