import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# todo
# comment colour_email() better, make docstring
# Make colour_extremity() have ham and spam labels above them in the gradient colours


def full_analysis_1_email(email: str, freq: pd.DataFrame, posterior_spam: float) -> str:
    output = f"There is a {round(posterior_spam * 100, 2)}% chance that this email is spam.\n" \
             f"Below is a colour-coded printout of the email according to the gradient:\n"

    output += colour_extremity() + "\n"
    output += colour_email(email, freq)
    return output


def colour_email(email: str, freq: pd.DataFrame) -> str:
    # Split the email string into lines using the newline character as the delimiter
    lines = email.split("\n")

    # Split each line into words, preserving capitalization and punctuation using regular expressions
    words = [re.findall(r"\w+[\.,!%'$]*\w*|\$?\d+|[^\w\s]", line) for line in lines]  # this took a while, couldn't reproduce it wihtout consulting a textbook tbh

    # Calculate the probability of each word being spam
    word_probs = []
    for line in words:
        word_line_probs = []
        for word in line:
            if word.lower() in freq["words"].values:
                spam_freq = freq.loc[freq["words"] == word.lower(), "spam"].iloc[0]
                ham_freq = freq.loc[freq["words"] == word.lower(), "ham"].iloc[0]
                word_prob = spam_freq / (spam_freq + ham_freq)
                word_line_probs.append(word_prob)
            else:
                # If the word is not in the frequency table, assume it's equally likely to be spam or ham
                word_line_probs.append(0.5)
        word_probs.append(word_line_probs)

    # Define a colour map for the heatmap
    cmap = plt.cm.get_cmap("RdYlBu_r")

    # Create a list to store the formatted words
    formatted_words = []

    # Format each word with the appropriate colour
    for line_index, line in enumerate(words):
        formatted_line = []
        for word_index, word in enumerate(line):
            if re.match(r"\w+", word):
                # If the word is alphanumeric, colour it based on its spam probability
                current_colour = cmap(word_probs[line_index][word_index])
                current_colour_code = f"\033[38;2;{int(current_colour[0] * 255)};{int(current_colour[1] * 255)};{int(current_colour[2] * 255)}m"
                formatted_word = f"{current_colour_code}{word}"
            else:
                # If the word is not alphanumeric, don't colour it
                formatted_word = word
            formatted_line.append(formatted_word)
        formatted_words.append(" ".join(formatted_line))

    # Combine the formatted lines into a single string with newline characters separating lines
    formatted_email = "\n".join(formatted_words)

    # Return the formatted email string with a colour escape code to reset colours for later terminal use
    return formatted_email + "\033[0m"


def colour_extremity(label: bool = True, length: int = 72) -> str:
    """
    Makes gradient colourmap for how extreme a certain colour means
    :param label: whether "ham" and "spam" should be displayed on top
    :param length: how many characters long it should be
    :return: a string with the appropriate colours to be printed in the command line
    """
    if not label:
        output = ""
    else:
        output = "Likely Ham" + " "*(length - len("Likely Spam") - len("Likely Ham")) + "Likely Spam" + "\n"
    # output = ""
    # Define the colour map
    cmap = plt.cm.get_cmap("RdYlBu_r")

    # Calculate the colour at each block
    colours = [cmap(x) for x in np.linspace(0, 1, length)]
    # Display the colour blocks
    for colour in colours:
        r, g, b = [int(x * 255) for x in colour][:-1]
        colour_code = f"\033[38;2;{r};{g};{b}m"
        output += colour_code + "█" + "\033[0m"
    return output + "\033[0m"

