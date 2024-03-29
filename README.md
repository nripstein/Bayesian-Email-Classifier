

<!--$$
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
$$-->

# Naive Bayes Spam Email Classifier

<div align="center">
    <a href="https://numpy.org"><img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" /></a>
    <a href="https://pandas.pydata.org"><img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" /></a>
    <a href="https://matplotlib.org"><img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" /></a>
</div>
<br>




- [TLDR](https://github.com/nripstein/Bayesian-Email-Classifier#tldr)
- [Why I Started This Project](https://github.com/nripstein/Bayesian-Email-Classifier#why-i-started-this-project)
- [How to Use](https://github.com/nripstein/Bayesian-Email-Classifier#how-to-use)
- [Have an Email? Check if it's Spam](https://github.com/nripstein/Bayesian-Email-Classifier#have-an-email-check-if-its-spam)
- [Classifier Accuracy Analysis and Visualizations](https://github.com/nripstein/Bayesian-Email-Classifier#classifier-accuracy-analysis-and-visualizations)
- [Mathematical Steps](https://github.com/nripstein/Bayesian-Email-Classifier#mathematical-steps)
- [Future Directions](https://github.com/nripstein/Bayesian-Email-Classifier#classifier-accuracy-analysis-and-visualizations)

## TLDR

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This program classifies emails as spam or ham.  It uses a naïve bayes classifier algorithm, which is a machine learning algorithm.  It does not use any machine learning libraries, rather, I designed and customized the algorithm using Bayesian inference.

## Why I Started This Project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In a Bayesian statistic class, I learned about the naive Bayes classifier and its applications in text classification. I was intrigued by the algorithm's simplicity and effectiveness, so I decided to implement it from scratch in Python without using any machine learning libraries. I wanted to challenge myself to understand the underlying math and logic of the algorithm, and to gain hands-on experience in building machine learning models from scratch.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Currently, I'm also learning how to create machine learning algorithms using neural networks in TensorFlow. I hope to come back to this project at some point to compare the performance of the naive Bayes classifier with a neural network classification on the same task. This project serves as a stepping stone for me to explore the world of machine learning and deepen my understanding of the algorithms behind it.

## How to use:
1. Download the repository and extract the files to your desired directory.
2. [Download the Waterloo Public Corpus Dataset from Kaggle](https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset.)
3. Create a folder called "data" in the root directory of the repository.
4. Move the downloaded dataset file into the "data" folder.
5. Run main.py to start the program.
6. Customize use of the program according to the comments at the bottom of the main.py file.  
Note: you will need to retrain the model, which may take some time, because the training data is stored in a csv file too large to upload to github

## Have an Email? Check if it's Spam
1. Train the model first by following the instructions in the "How to Use" section.
2. Open the to_classify.txt file located in the main folder of the repository.
3. Paste the email you want to classify into the to_classify.txt file.
4. Save the to_classify.txt file and close it.
5. Run main.py to classify and colour-code the email

The console will output the posterior probability of the email being spam, and will output the email with words colour coded according to their probability of coming from a spam email.  Examples below.

It is important to remember that the model was trained on a specific dataset, and as such, it may not be accurate in classifying all types of emails. The model was trained on the Waterloo Public Corpus dataset, which includes a large collection of emails from different sources. It is important to note that the model is not personalized to you or any individual, and as such, its accuracy may vary depending on the email content.  For instance, the training dataset labels almost all French emails as spam, so the words the model thinks are most likely to be spam are all in French.

Examples:

<p align="center">
  <img src="https://user-images.githubusercontent.com/98430636/220826883-b6ffc540-60f0-4153-9acd-1a617d8426b5.png" alt="spam" width="90%">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/98430636/220826890-44fc7199-8d15-4b77-93d0-d55be015752c.png" alt="ham" width="90%">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/98430636/220827124-0b710823-ec55-441d-acaa-142dca24c2a8.png" alt="prob spam" width="90%">
</p>



## Classifier Accuracy Analysis and Visualizations
To evaluate the performance of the Naive Bayes spam email classifier, I trained it on 15,000 emails and tested it on 2,000. The overall accuracy of the model was 92.05%, which seems like a promising result. The confusion matrix below shows the number of true positives, true negatives, false positives, and false negatives for the test set.


### Confusion Matrix

<p align="center">
  <img src="https://user-images.githubusercontent.com/98430636/220796903-e92a632e-a977-4ffc-bb74-c94f31e747b9.png" alt="confusion matrix" width="70%">
</p>

The data from this confusion matrix can be used to determine the <u>sensitivity</u> and <u>specificity</u> of my algorithm. Sensitivity and specificity are two important measures used to evaluate the performance of a binary classification algorithm. Sensitivity measures the proportion of true positive cases that are correctly identified as positive, while specificity measures the proportion of true negative cases that are correctly identified as negative.

For my spam email classifier, I calculated the sensitivity and specificity based on the confusion matrix generated from testing the algorithm on 2,000 emails. We found that the most likely specificity of my algorithm was 99.8%, with a 95% confidence interval of 99.7% to 99.9%. The most likely sensitivity of my algorithm was 86.1%, with a 95% confidence interval of 84.3% to 87.9%.

To visualize the probability distribution of sensitivity and specificity, I generated probability distribution functions for each measure. These graphs  provide a way to visualize the estimattion for the a particular value being the true sensitivity or specificity. The probability distribution functions for sensitivity and specificity are shown below:

### Sensitivity and Specificity Probability Distribution Functions

<p align="center">
    <img src="https://user-images.githubusercontent.com/98430636/220942351-de3ad74b-f338-4678-bacd-544e209d9e61.png" alt="sens, spec" width="70%">
</p>



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As you can see, the probability distribution function for sensitivity is extremley tightly distributed near 100%. This indicates that the model is very good at identifying non-spam emails, with a very low false positive rate. On the other hand, the specificity pdf shows a wider distribution with a lower mode, indicating that the model is not as good at identifying spam emails, and the degree of certainty in the sensitivity it lower. However, the 95% confidence interval for sensitivity still shows a relatively high accuracy rate, ranging from 84.3% to 87.9%. I would ideally like to improve the model so that the lower end of the sensitivity 95% confidence interval is at least 90%.

## Word Clouds
### Most Common Spam Words
<p align="center">
    <img src="https://user-images.githubusercontent.com/98430636/220796954-f5bfe2d8-a98b-40f3-91ae-5c2b37af3dda.png" alt="spam ratio" width="90%">
</p>


### Most Common Ham Words
<p align="center">
    <img src="https://user-images.githubusercontent.com/98430636/220796981-ae2880dc-37cd-4715-9413-91203022eabc.png" alt="ham ratio" width="90%">
</p>

There are lots of French words which are common in the spam corpus, but very rare in the ham corpus.  This indicates that French emails are very likely spam in the training set.
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



## Future Directions
There are a few areas where the spam email classifier could be improved. Some of the possible areas for improvement include:

1. Handling spam emails made up of one long word: Currently, the model is not equipped to handle spam emails that contain a single long word that it has not seen before. In future versions of the model, I plan to develop a mechanism to handle such words and improve the model's accuracy.

2. Conducting more analysis on accuracy: While the model has shown promising results, it is clear that there is room for improvement in terms of accuracy. In order to figure out how to best improve the model, I hope to conduct more analysis on the accuracy.

3. Determining the optimal probability threshold: The current version of the model classifies an email as spam or not spam based on a probability threshold of 0.95. In future versions of the model, I plan to conduct a more in-depth analysis to determine the optimal probability threshold that will provide the best classification results.

By addressing these issues, I hope to improve the accuracy and effectiveness of the spam email classifier.


