import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
tqdm.pandas()




def generate_wordclouds(freq: pd.DataFrame):
    os.chdir("/Users/NoahRipstein/PycharmProjects/Bayes email 2/Word Clouds")
    freq = clean_common_words(freq)
    # Create separate dataframes for "ham" and "spam" frequencies
    ham_df = freq[["words", "ham"]].set_index("words")
    spam_df = freq[["words", "spam"]].set_index("words")
    print(ham_df)
    # Generate word clouds
    ham_wordcloud = WordCloud().generate_from_frequencies(ham_df.to_dict()["ham"])
    # print(ham_df.to_dict()["ham"])
    ham_wordcloud.to_file("ham_wordcloud.png")
    spam_wordcloud = WordCloud().generate_from_frequencies(spam_df.to_dict()["spam"])
    spam_wordcloud.to_file("spam_wordcloud.png")
    # Display word clouds
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ham_wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Ham Word Cloud")

    plt.subplot(1, 2, 2)
    plt.imshow(spam_wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Spam Word Cloud")

    df2 = freq["spam"] / freq["ham"]
    ls = df2.values
    # print(df2)
    # print(ls)

    # Calculate relative frequencies of each word in "spam" compared to "ham"
    # relative_frequencies = {}
    # for word in dataframe["words"]:
    #     if word not in relative_frequencies:
    #         spam_freq = spam_df.loc[word].values[0]
    #         ham_freq = ham_df.loc[word].values[0]
    #         if ham_freq == 0:
    #             relative_frequencies[word] = spam_freq
    #         else:
    #             relative_frequencies[word] = spam_freq / ham_freq
    # print("done")
    # return relative_frequencies


def clean_common_words(freq: pd.DataFrame) -> pd.DataFrame:
    drop_list = ["the", "a", "to", "and", "of", "is"]
    return freq.loc[~freq["words"].isin(drop_list)]


def spam_ratio(freq: pd.DataFrame):
    freq = freq.loc[(freq["ham"] >= 1) & (freq["spam"] >= 1)]
    freq = freq.copy()

    freq["ratio"] = freq["spam"] / freq["ham"]

    print("done f")
    # freq = freq.drop(index=freq.loc[f].index)
    # print(freq)
    freq = freq.sort_values(by="ratio", ascending=False)
    # print(freq.head(20))
    freq = freq[["words", "ratio"]].set_index("words")
    freq = freq.sort_values(by="ratio")
    # print(freq.head(20))
    my_dict = freq.to_dict()["ratio"]

    ratio_wordcloud = WordCloud().generate_from_frequencies(my_dict)
    ratio_wordcloud.to_file("spam_ratio.png")








