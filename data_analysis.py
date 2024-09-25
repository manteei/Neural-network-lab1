import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

df = pd.read_csv('news.csv')

############################
#1. Распределение количества
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', data=df)
plt.title('Распределение реальных и фейковых новостей')
plt.xlabel('Метка (real/fake)')
plt.ylabel('Количество')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# ##############################
# #2. Распределение длины текста
df['text_length'] = df['text'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(df, x='text_length', hue='label', kde=True)
plt.title('Распределение длины текста новостей')
plt.xlabel('Длина текста')
plt.ylabel('Частота')

plt.show()

#################################
#3. Распределение длины заголовка
df['title_length'] = df['title'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(df, x='title_length', hue='label', kde=True)
plt.title('Распределение длины заголовков')
plt.xlabel('Длина заголовка')
plt.ylabel('Частота')
plt.show()

# #############################
# #4. Наиболее популярные слова
fake_text = " ".join(df[df['label'] == 'FAKE']['text'])
wordcloud_fake = WordCloud(width=800, height=400).generate(fake_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title('Облако слов для фейковых новостей')
plt.axis('off')
plt.show()

real_text = " ".join(df[df['label'] == 'REAL']['text'])
wordcloud_real = WordCloud(width=800, height=400).generate(real_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.title('Облако слов для реальных новостей')
plt.axis('off')
plt.show()

# ################################
#5 Доля уникальных слов в тексте
df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.split())))

plt.figure(figsize=(10, 6))
sns.histplot(df, x='unique_word_count', hue='label', kde=True)
plt.title('Распределение количества уникальных слов в новостях')
plt.xlabel('Количество уникальных слов')
plt.ylabel('Частота')
plt.xticks(ticks=range(0, df['unique_word_count'].max() + 1, 100), rotation=90)
plt.show()

# ##############################################
# #6 Соотношение уникальных слов к общему числу слов
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['lexical_diversity'] = df['unique_word_count'] / df['word_count']

plt.figure(figsize=(10, 6))
sns.histplot(df, x='lexical_diversity', hue='label', kde=True)
plt.title('Распределение лексической насыщенности новостей')
plt.xlabel('Лексическая насыщенность (уникальные слова / общие слова)')
plt.ylabel('Частота')
plt.show()

# ###############################################
# #7. Самые частые словосочетания
def get_top_n_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

df_fake = df[df['label'] == 'FAKE']
df_real = df[df['label'] == 'REAL']

fake_bigrams = get_top_n_bigrams(df_fake['text'], 20)
fake_bigram_df = pd.DataFrame(fake_bigrams, columns=['bigram', 'frequency'])

real_bigrams = get_top_n_bigrams(df_real['text'], 20)
real_bigram_df = pd.DataFrame(real_bigrams, columns=['bigram', 'frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(x='frequency', y='bigram', data=fake_bigram_df)
plt.title('Топ-20 самых частых биграмм в фейковых новостях')
plt.xlabel('Частота')
plt.ylabel('Биграммы')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='frequency', y='bigram', data=real_bigram_df)
plt.title('Топ-20 самых частых биграмм в реальных новостях')
plt.xlabel('Частота')
plt.ylabel('Биграммы')
plt.show()

# #####################################
# # 8. Распределение тональности текста
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['sentiment'] = df['text'].apply(get_sentiment)
plt.figure(figsize=(10, 6))
sns.histplot(df, x='sentiment', hue='label', kde=True)
plt.title('Распределение тональности новостей')
plt.xlabel('Тональность')
plt.ylabel('Частота')
plt.show()
