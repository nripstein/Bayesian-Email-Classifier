<!-- MathJax script -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<!-- End of MathJax script -->

# Bayes-email-4

## TLDR

This program classifies emails as spam or not spam.  It uses a naïve bayes classifier algorithm, which is a machine learning algorithm.  It does not use any machine learning libraries, rather, I designed and customized the algorithm using Bayesian inference.

## How to use:

## Mathematical steps:

Preparation for training:
-	The body section of each email’s is converted to a set of words, which is added to a column called “word_set” using the string_to_word_set() function.
-	The dataset is split into a training and testing data set.

Training the data:
-	A new data frame is created which contains the frequency at which each word in the training data set appears in the “spam” and “ham” category.

Classifying the data:
-	The Bayesian posterior probability that an email is spam is determined according to the following (basic) procedure:

P(spam|evidence) = prior * likelihood of all words

Likelihoods: P(words|category)

$$P(words|spam) = \prod_{i=1}^{n} \frac{\textrm{freq of word}_i \textrm{in spam}}{\textrm{ham words}}$$

$$P(words|ham) = \prod_{i=1}^{n} \frac{\textrm{freq of word}_i \textrm{in ham}}{\textrm{ham words}}$$

Posteriors:

$$ P(H1|D) = \frac{P(D|H1)P(H1)}{P(D|H1)P(H1) + P(D|H2)P(H2)} $$

$$ P(spam|words) = \frac{P(words|span)P(spam)}{P(words|spam)P(spam) + P(words|ham)P(ham)} $$

$$ P(spam|words) \frac{\prod_{i=1}^{n} \frac{\textrm{freq of word}_i \textrm{in spam}}{\textrm{ham words}}P(spam)}{P(words|spam)P(spam) + P(words|ham)P(ham)}$$

Deviations from standard procedure:
-	If a word appears in “spam”
-	
-	email in the testing data set is 

![image](https://user-images.githubusercontent.com/98430636/219830713-955e4862-a03e-414b-a746-72d83dea6699.png)
