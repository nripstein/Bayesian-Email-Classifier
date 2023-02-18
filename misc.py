import pandas as pd
import numpy as np
import os
import string
from typing import Union, Callable

import fileIO
import accuracy_analysis

# progress bar stuff
from tqdm import tqdm
tqdm.pandas()


# todo
# make posterior function better so it can better account for words it gets


def timer(func: Callable) -> Callable:
    """decorator to measure how long a function takes to run"""
    import time

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds to run")
        return result
    return wrapper


def add_to_freq_df_old(freq: pd.DataFrame, category: str, words: set[str], key: str = "words") -> pd.DataFrame:
    """
    This is no longer in use.  It is much slower than the current version because it iterates instead of using vectorization

    :param freq: dataframe of frequencies of each word (or, specifically, each key)
    :param category: the category the current message is part of (like "spam")
    :param words: a set of all the words in the message
    :param key: the name of the column with frequencies in the other cols
    :return: updated dataframe with frequencies of words in each category
    """
    if category not in freq.columns:
        raise IndexError(f"\nCategory must be in frequency dataframe. {category} not in \n{freq.columns}\n"
                         f"Categories are normally spam or ham\n")

    for word in words:
        if freq[key].isin([word]).any():  # if word in freq
            freq.loc[freq[key] == word, category] += 1  # add 1 to the right category
        else:  # initialize row with word where it's category = 1, everything else is 0
            new_row: pd.DataFrame = pd.DataFrame({key: [word], category: [1]})  # make df with the key and right cat = 1
            for column in freq.columns:  # set rest of values for columns to 0 in new df
                if column not in new_row.columns:
                    new_row[column] = 0
            freq: pd.DataFrame = pd.concat([freq, new_row], ignore_index=True, sort=False)  # combine them
    return freq


def add_to_freq_df(freq: pd.DataFrame, category: str, words: set[str], key: str = "words") -> pd.DataFrame:
    """
    :param freq: dataframe of frequencies of each word (or, specifically, each key)
    :param category: the category the current message is part of (like "spam")
    :param words: a set of all the words in the message
    :param key: the name of the column with frequencies in the other cols
    :return: updated dataframe with frequencies of words in each category
    """
    if category not in freq.columns:
        raise IndexError(f"\nCategory must be in frequency dataframe. {category} not in \n{freq.columns}\n"
                         f"Categories are normally spam or ham\n")

    # create boolean mask of rows that contain words in the set
    mask = freq[key].isin(words)

    # update appropriate cells in DataFrame using boolean mask
    freq.loc[mask, category] += 1

    # add new rows for words not already in the DataFrame
    # make new dataframe with words for each row that's unlabelled with right row 1 and others at 0
    new_rows = pd.DataFrame({key: list(words - set(freq[key])), category: 1})
    for column in freq.columns:
        if column not in new_rows.columns:
            new_rows[column] = 0
    freq = pd.concat([freq, new_rows], ignore_index=True, sort=False)

    return freq




@timer
def train_n_rows(trainer: pd.DataFrame, freq: pd.DataFrame = None, num_rows: int = 10,
                 all_rows: bool = False, progress_bar: bool = True) -> pd.DataFrame:
    """

    :param trainer: dataframe with emails we're training
    :param freq: dataframe with frequencies of wors
    :param num_rows: number of rows to train
    :param all_rows: whether we want to train whole dataset
    :param progress_bar: if you want progress bar displayed
    :return: frequency dataset with new training data
    """
    if freq is None:
        freq = pd.DataFrame({"words": [], "spam": [], "ham": []})

    label_dict = {1: "spam", 0: "ham"}
    if all_rows:
        num_rows = len(trainer.index)

    if progress_bar:
        for row_index in tqdm(range(num_rows), desc="Training emails", ncols=100):
            row = trainer.iloc[row_index]
            label = label_dict[row["Label"]]
            freq = add_to_freq_df(freq, category=label, words=row["word_set"])
    else:
        for row_index in range(num_rows):
            row = trainer.iloc[row_index]
            label = label_dict[row["Label"]]
            freq = add_to_freq_df(freq, category=label, words=row["word_set"])
    return freq


def one_posterior(word_set: set[str], freq: pd.DataFrame, prior_spam: float = 0.5, thresh: float = 0.95) -> tuple[float, int]:
    """
    Finds the posterior probability that an email is spam
    :param word_set: set of words in the email. from word_set column of trainer dataframe
    :param freq: frequency of each word in each category dataframe
    :param prior_spam: prior probability that an email is spam
    :param thresh: if an email has greater than a thresh probability of being spam, then we classify it as span
    :return: probability of it being spam, classification
    """
    # need to deal w words in word set but not freq

    # laplace smoothing to deal with words with only 1
    smoothed = freq.copy()
    smoothed["spam"] = smoothed["spam"] + 1
    smoothed["ham"] = smoothed["ham"] + 1

    # freq w right words
    found_words = smoothed.loc[smoothed["words"].isin(word_set)]

    if found_words.empty:  # if the frequency dataset has no words in common with email, we say p(spam|D) = 0.5
        return 0.5, int(False)

    likelihood_spam_individual = found_words["spam"].to_numpy() / found_words["spam"].sum()
    likelihood_ham_individual = found_words["ham"].to_numpy() / found_words["ham"].sum()

    # likelihood_spam = np.prod(likelihood_spam_individual)
    # likelihood_ham = np.prod(likelihood_ham_individual)

    # likelihood_spam = np.exp(np.sum(np.log(likelihood_spam_individual)))
    # likelihood_ham = np.exp(np.sum(np.log(likelihood_ham_individual)))

    likelihood_ratio = np.exp(np.sum(np.log((likelihood_ham_individual / likelihood_spam_individual))))

    if prior_spam == 0.5 or prior_spam is None:
        posterior = 1 / (1 + likelihood_ratio)
        return posterior, int(posterior > thresh)
    else:
        posterior = prior_spam / (prior_spam + likelihood_ratio * (1 - prior_spam))
        return posterior, int(posterior > thresh)


def one_posterior3(word_set: set[str], freq: pd.DataFrame, prior_spam: float = 0.5, thresh: float = 0.95) \
        -> tuple[float, int]:
    """
    Finds the posterior probability that an email is spam
    :param word_set: set of words in the email. from word_set column of trainer dataframe
    :param freq: frequency of each word in each category dataframe
    :param prior_spam: prior probability that an email is spam
    :param thresh: if an email has greater than a thresh probability of being spam, then we classify it as span
    :return: probability of it being spam, classification
    """
    # need to deal w words in word set but not freq

    # update appropriate cells in DataFrame using boolean mask
    freq.loc[freq["spam"] == 0, "spam"] = 0.5
    freq.loc[freq["ham"] == 0, "ham"] = 0.01  # if there's nothing in ham, set ham = 0.01

    # freq w right words
    found_words = freq.loc[freq["words"].isin(word_set)]

    # non_words = smoothed.loc[~smoothed["words"].isin(word_set)]
    # print(non_words)

    if found_words.empty:  # if the frequency dataset has no words in common with email, we say p(spam|D) = 0.5
        return 0.5, int(False)

    likelihood_spam_individual = found_words["spam"].values / found_words["spam"].sum()
    likelihood_ham_individual = found_words["ham"].values / found_words["ham"].sum()

    # likelihood_spam = np.prod(likelihood_spam_individual)
    # likelihood_ham = np.prod(likelihood_ham_individual)

    # likelihood_spam = np.exp(np.sum(np.log(likelihood_spam_individual)))
    # likelihood_ham = np.exp(np.sum(np.log(likelihood_ham_individual)))

    likelihood_ratio = np.exp(np.sum(np.log((likelihood_ham_individual / likelihood_spam_individual))))
    # likelihood_ratio = np.prod(likelihood_ham_individual / likelihood_spam_individual)
    if prior_spam == 0.5 or prior_spam is None:
        posterior = 1 / (1 + likelihood_ratio)
        return posterior, int(posterior > thresh)
    else:
        posterior = prior_spam / (prior_spam + likelihood_ratio * (1 - prior_spam))
        return posterior, int(posterior > thresh)


def one_posterior3_optimized(word_set: set[str], freq: pd.DataFrame, prior_spam: float = 0.5, thresh: float = 0.975) \
        -> tuple[float, int]:
    """
    Finds the posterior probability that an email is spam
    :param word_set: set of words in the email. from word_set column of trainer dataframe
    :param freq: frequency of each word in each category dataframe
    :param prior_spam: prior probability that an email is spam
    :param thresh: if an email has greater than a thresh probability of being spam, then we classify it as span
    :return: probability of it being spam, classification
    """
    # laplace smoothing to deal with words with only 1
    freq.loc[freq["spam"] == 0, "spam"] = 0.5
    freq.loc[freq["ham"] == 0, "ham"] = 0.01  # if there's nothing in ham, set ham = 0.01

    # freq w right words
    found_words = freq.loc[freq["words"].isin(word_set)]

    if found_words.empty:  # if the frequency dataset has no words in common with email, we say p(spam|D) = 0.5
        return 0.5, int(False)

    likelihood_spam_individual = found_words["spam"].values / found_words["spam"].sum()
    likelihood_ham_individual = found_words["ham"].values / found_words["ham"].sum()

    likelihood_ratio = likelihood_ham_individual.sum() / likelihood_spam_individual.sum()

    if prior_spam == 0.5 or prior_spam is None:
        posterior = 1 / (1 + likelihood_ratio)
        return posterior, int(posterior > thresh)
    else:
        posterior = prior_spam / (prior_spam + likelihood_ratio * (1 - prior_spam))
        return posterior, int(posterior > thresh)


def posterior_col(tester: pd.DataFrame, freq: pd.DataFrame, compare: bool = True,
                  progress_bar: bool = True) -> pd.DataFrame:
    """
    makes column with posterior number and column with classification
    :param tester: dataframe we're trying our predictions on
    :param freq: dataset with frequency of each word in each category that we trained
    :param compare: if true, makes column indicating whether classification was correct
    :param progress_bar: if you want to display a progress bar
    :return: tester with P(spam) and computer_label columns added
    """
    tester2 = tester.copy()
    if progress_bar:
        tester2[["P(spam)", "computed_label"]] = tester2["word_set"].progress_apply(lambda word_set: pd.Series(one_posterior3(word_set, freq)))  # should be without numbebr affter one_posterior
    else:
        tester2[["P(spam)", "computed_label"]] = tester2["word_set"].apply(lambda word_set: pd.Series(one_posterior3(word_set, freq)))
    if compare:
        tester2["correct?"] = tester2["computed_label"] == tester2["Label"]
    return tester2


def compare_score(tester: pd.DataFrame):
    """
    adds column to tester dataframe which has already been classified to see which classifications were correct
    :param tester: dataframe with emails that's been classified by the program already
    :return:
    """
    tester2 = tester.copy()
    tester2["correct?"] = tester2["computed_label"] == tester2["Label"]
    return tester2


def string_to_word_set(body: str = None) -> set[str]:
    """
    makes a set of all words in an email given the body as a string
    :param body: body of the email
    :return: complete set of words in the email
    """
    # $ is removed here. other things to include or exclude could be helpful. could also do something for super long words
    for char in set(string.punctuation).union({"\n", "\t"}):
        body = body.replace(char, " ")

    raw_body = set(body.split(" "))  # makes a set of all the words
    lowercase_body = set(map(lambda x: x.lower(), raw_body))  # makes all words in the set lowercase
    return set(filter(lambda x: x.strip(), lowercase_body))  # mb can optimize? but wb


@timer
def add_word_set_col(full_df: pd.DataFrame, body_col_name: str = "Body") -> pd.DataFrame:
    """Adds "word_set" column to dataframe with word set of the body_col_name = "Body" column"""
    full_df["word_set"] = full_df.apply(lambda row: string_to_word_set(row[body_col_name]), axis=1)
    return full_df


if __name__ == '__main__':
    # corpus = fileIO.load_emails()
    # fileIO.save_formatted_raw_emails(corpus)
    corpus = fileIO.load_emails(file_handle="formatted_corpus", file_extension="pickle", reformat=False)

    corpus = corpus.head(16000)

    with_word_set = add_word_set_col(corpus)

    # training_df = with_word_set.sample(frac=0.9, random_state=1)
    # testing_df = with_word_set.drop(with_word_set.index)

    freq2 = fileIO.load_freq()
    # freq2 = train_n_rows(df2, num_rows=15000)
    # fileIO.save_freq(freq2)

    tester = with_word_set.tail(1000)

    classified = posterior_col(tester, freq2)
    # comparison = compare_score(classified)
    print(classified["correct?"].value_counts()[True] / (
                classified["correct?"].value_counts()[False] + classified["correct?"].value_counts()[True]))
    print(classified.columns)
    print()

    full, wrong_only = accuracy_analysis.error_type_row(classified)
    print(type(wrong_only))

    print(wrong_only.loc[:, ["error_type", "P(spam)"]])
    print(wrong_only.groupby("error_type").size())

    # print(wrong_only[wrong_only["error_type"] != "Correct Answer"])

