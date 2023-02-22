

<!--$$
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
$$-->

# Naive Bayes Spam Email Classifier

- [TLDR](https://github.com/nripstein/Bayes-email-4/edit/main/README.md#mathematical-steps)
- [How to Use](https://github.com/nripstein/Bayes-email-4/edit/main/README.md#mathematical-steps)
- [Mathematical Steps](https://github.com/nripstein/Bayes-email-4/edit/main/README.md#mathematical-steps)
- [Have an Email? Check if it's Spam](https://github.com/nripstein/Bayes-email-4/edit/main/README.md#have-an-email-check-if-its-spam)
- [Classifier Accuracy Analysis and Visualizations](https://github.com/nripstein/Bayesian-Email-Classifier/edit/main/README.md#classifier-accuracy-analysis-and-visualizations)
- [Future Directions](https://github.com/nripstein/Bayes-email-4/edit/main/README.md#future-directions)

## TLDR

This program classifies emails as spam or not spam.  It uses a naïve bayes classifier algorithm, which is a machine learning algorithm.  It does not use any machine learning libraries, rather, I designed and customized the algorithm using Bayesian inference.

## How to use:
This section is incomplete

## Mathematical steps:

Preparation for training:
-	The body section of each email’s is converted to a set of words, which is added to a column called “word_set” using the string_to_word_set() function.
-	The dataset is split into a training and testing data set.

Training the data:
-	A new data frame is created which contains the frequency at which each word in the training data set appears in the “spam” and “ham” category.

Classifying the data:
-	The Bayesian posterior probability that an email is spam is determined according to the following (basic) procedure:

### Bayesian posterior calculation

Hypotheses:  
H1: The email is spam
H2: The email is ham

Prior probabilities: P(category)  
P(spam) and P(ham). We treat these as P(spam) = P(ham) = 0.5. In the strictest mathematical sense, P(spam) should be the overall frequency of spam emails, but I want to reduce bias, and the training dataset has much more spam than a normal email adress.

Likelihood calculations: P(words|category)  
$$P(\textrm{words}|\textrm{spam}) = \prod_{i=1}^{n} \frac{\textrm{freq of word}_i \textrm{in spam}}{\textrm{ham words}}$$

$$P(\textrm{words}|\textrm{ham}) = \prod_{i=1}^{n} \frac{\textrm{freq of word}_i \textrm{in ham}}{\textrm{ham words}}$$

Posteriors:

$$ P(H1|D) = \frac{P(D|H1)P(H1)}{P(D|H1)P(H1) + P(D|H2)P(H2)} $$

$$ P(\textrm{spam}|\textrm{words}) = \frac{P(\textrm{words}|\textrm{spam})P(\textrm{spam})}{P(\textrm{words}|\textrm{spam})P(\textrm{spam}) + P(\textrm{words}|\textrm{ham})P(\textrm{ham})} $$

Deviations from standard procedure:
-	If, when doing a likelihood calculstion, a word appears in “spam” but not "ham," then it is treated as if it's been in ham 0.5 times.
-	If a word appears in "ham" but not "spam," then it is treated as if it has appeared 0.01 tines

The posterior probability of each email being spam is computed.  If it's determined that there's a greater than 90% probability of the email being spam, then it is classified as spam


## Have an Email? Check if it's Spam
This section is incomplete.  Features coming soon
- I will impliment a feature where you can paste an email into a txt file and the program will tell you the probability that it is spam, and print out the words colour coded according to a heatmap for how likely it is to be spam

## Classifier Accuracy Analysis and Visualizations
This section is incomplete

I intend to show:
- confusion matrix
- ratio wordclouds


## Future Directions
This section is incomplete
- If an email is one long word, my algorithm doesn't really know how to treat it. I'd like to do something about that

![image](https://user-images.githubusercontent.com/98430636/219830713-955e4862-a03e-414b-a746-72d83dea6699.png)
