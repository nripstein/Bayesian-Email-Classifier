import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# todo
# comment colour_email() better, make docstring
# make docstring for colour_extremity()

def colour_email(email: str, freq: pd.DataFrame) -> str:
    # Split the email string into lines using the newline character as the delimiter
    lines = email.split("\n")

    # Split each line into words, preserving capitalization and punctuation
    words = [re.findall(r"\w+[\.,!%'$]*\w*|\$?\d+|[^\w\s]", line) for line in lines]  # this took a while, probably couldn't reproduce it

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

    # Define a color map for the heatmap
    cmap = plt.cm.get_cmap("RdYlBu_r")

    # Create a list to store the formatted words
    formatted_words = []

    # Format each word with the appropriate color
    for line_index, line in enumerate(words):
        formatted_line = []
        for word_index, word in enumerate(line):
            if re.match(r"\w+", word):
                # If the word is alphanumeric, color it based on its spam probability
                current_colour = cmap(word_probs[line_index][word_index])
                current_colour_code = f"\033[38;2;{int(current_colour[0] * 255)};{int(current_colour[1] * 255)};{int(current_colour[2] * 255)}m"
                formatted_word = f"{current_colour_code}{word}"
            else:
                # If the word is not alphanumeric, don't color it
                formatted_word = word
            formatted_line.append(formatted_word)
        formatted_words.append(" ".join(formatted_line))

    # Combine the formatted lines into a single string with newline characters separating lines
    formatted_email = "\n".join(formatted_words)

    # Return the formatted email string with a colour escape code to reset colours for later terminal use
    return formatted_email + "\033[0m"


def colour_extremity(label: bool = True, length: int = 50) -> str:
    if not label:
        output = ""
    else:
        output = "Ham" + " "*(length - len("spam") - len("ham")) + "Spam" + "\n"

    # Define the color map
    cmap = plt.cm.get_cmap("RdYlBu_r")

    # Calculate the color at each block
    colors = [cmap(x) for x in np.linspace(0, 1, length)]

    # Display the color blocks
    for color in colors:
        # print([int(x * 255) for x in color])
        r, g, b = [int(x * 255) for x in color][:-1]
        color_code = f"\033[38;2;{r};{g};{b}m"
        output += color_code + "█" + "\033[0m"
        # print(color_code + "█" + "\033[0m", end="")
    return output + "\033[0m"

