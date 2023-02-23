import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os


def generate_wordclouds(freq: pd.DataFrame):
    # probably not useful because it's not ratio
    os.chdir("/Users/NoahRipstein/PycharmProjects/Bayes email 2/visualizations")
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


def ratio_cloud(freq: pd.DataFrame) -> None:
    os.chdir("/Users/NoahRipstein/PycharmProjects/Bayes email 2/visualizations")
    freq = freq.loc[(freq["ham"] >= 1) & (freq["spam"] >= 1)]
    freq = freq.copy()

    freq["spam_ratio"] = freq["spam"] / freq["ham"]
    freq_export_spam = freq[["words", "spam_ratio"]].set_index("words")
    spam_dict = freq_export_spam.to_dict()["spam_ratio"]

    spam_ratio_wordcloud = WordCloud(width=1600//4, height=800//4, scale=4).generate_from_frequencies(spam_dict)
    spam_ratio_wordcloud.to_file("spam_ratio.png")

    freq["ham_ratio"] = freq["ham"] / freq["spam"]
    freq_export_ham = freq[["words", "ham_ratio"]].set_index("words")
    ham_dict = freq_export_ham.to_dict()["ham_ratio"]

    ham_ratio_wordcloud = WordCloud(width=1600//4, height=800//4, scale=4).generate_from_frequencies(ham_dict)
    ham_ratio_wordcloud.to_file("ham_ratio.png")
