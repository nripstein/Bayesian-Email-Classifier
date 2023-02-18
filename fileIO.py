import pandas as pd
import os

from misc import timer

# dataset from
# https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset?resource=download


@timer
def load_emails(directory: str = "/Users/NoahRipstein/PycharmProjects/Bayes email 2/data",
                file_handle: str = "2007_Public_Corpus", reformat: bool = True,
                file_extension: str = "csv") -> pd.DataFrame:

    def reformat_corpus(df: pd.DataFrame) -> pd.DataFrame:
        """
        used by load_emails() to format like spamassassin
        :param df: raw corpus dataframe
        :return: formatted like spamassassin
        """
        reformatted_corpus = df[["message", "label"]]
        reformatted_corpus = reformatted_corpus.rename(columns={"message": "Body", "label": "Label"})
        return reformatted_corpus

    os.chdir(directory)
    # corpus_in = pd.read_csv(file_handle)
    if file_extension == "csv":
        corpus_in = pd.read_csv(f"{file_handle}.{file_extension}")
    elif file_extension in ("pickle", "pkl"):
        corpus_in = pd.read_pickle(f"{file_handle}.{file_extension}")
    else:
        raise IOError(f'\nfile type not supported. Only {("csv", "pickle", "pkl")} supported')

    if reformat:
        corpus_in = corpus_in.dropna(subset=["message"])  # get rid of rows with NA in Body
        return reformat_corpus(corpus_in)
    else:
        return corpus_in


def save_formatted_raw_emails(corpus: pd.DataFrame, file_extension: str = "pickle",
                              directory: str = "/Users/NoahRipstein/PycharmProjects/Bayes email 2/data",
                              file_handle: str = "formatted_corpus") -> None:
    supported_file_extensions = ("csv", "pickle", "pkl")
    if file_extension not in supported_file_extensions:
        raise IOError(f"\nfile type not supported. Only {supported_file_extensions} supported")

    if file_extension == "csv":
        corpus.to_csv(f"{file_handle}.{file_extension}", index=False)
    elif file_extension in ("pickle", "pkl"):
        corpus.to_pickle(f"{file_handle}.{file_extension}")


def save_freq(freq: pd.DataFrame, file_extension: str = "csv",
              directory: str = "/Users/NoahRipstein/PycharmProjects/Bayes email 2/data",
              file_handle: str = "word_frequency") -> None:
    supported_file_extensions = ("csv", "pickle", "pkl")
    if file_extension not in supported_file_extensions:
        raise IOError(f"\nfile type not supported. Only {supported_file_extensions} supported")

    os.chdir(directory)
    if file_extension == "csv":
        freq.to_csv(f"{file_handle}.{file_extension}", index=False)
    elif file_extension in ("pickle", "pkl"):
        freq.to_pickle(f"{file_handle}.{file_extension}")


@timer
def load_freq(file_extension: str = "csv", directory: str = "/Users/NoahRipstein/PycharmProjects/Bayes email 2/data",
              file_handle: str = "word_frequency"):
    supported_file_extensions = ("csv", "pickle", "pkl")
    if file_extension not in supported_file_extensions:
        raise IOError(f"\nfile type not supported. Only {supported_file_extensions} supported")

    os.chdir(directory)

    if file_extension == "csv":
        return pd.read_csv(f"{file_handle}.{file_extension}")
    elif file_extension in ("pickle", "pkl"):
        return pd.read_pickle(f"{file_handle}.{file_extension}")





