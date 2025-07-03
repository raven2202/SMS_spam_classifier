# SMS_spam_classifier
This project is a machine learning-based SMS spam classifier that distinguishes between legitimate ("ham") and unsolicited ("spam") messages. It involves a complete pipeline from data cleaning and exploratory data analysis to text preprocessing, model building, evaluation, and improvement, with future plans for a web interface and deployment.
SMS Spam Classifier
This repository contains a machine learning project for classifying SMS messages as either "ham" (legitimate) or "spam". The project involves data cleaning, exploratory data analysis (EDA), text preprocessing, model building, evaluation, and improvement.

Table of Contents
SMS Spam Classifier

Table of Contents

Project Overview

Dataset

Project Structure

Steps Involved

1. Data Cleaning

2. Exploratory Data Analysis (EDA)

3. Text Preprocessing

4. Model Building

5. Evaluation

6. Improvement

7. Website (Future Work)

8. Deployment (Future Work)

Requirements

Usage

Results

Contributing

License

Project Overview
The goal of this project is to build a robust SMS spam classification model. We utilize various text processing techniques and machine learning algorithms to accurately distinguish between legitimate and unsolicited messages.

Dataset
The dataset used for this project is spam.csv. It contains SMS messages labeled as either "ham" or "spam".

Initial Data Snapshot (first 5 rows):

v1

v2

Unnamed: 2

Unnamed: 3

Unnamed: 4

ham

Go until jurong point, crazy.. Available only ...

NaN

NaN

NaN

ham

Ok lar... Joking wif u oni...

NaN

NaN

NaN

spam

Free entry in 2 a wkly comp to win FA Cup fina...

NaN

NaN

NaN

ham

U dun say so early hor... U c already then say...

NaN

NaN

NaN

ham

Nah I don't think he goes to usf, he lives aro...

NaN

NaN

NaN

Dataset Shape: (5572 rows, 5 columns)

Project Structure
sms_spam_classifier.ipynb: The main Jupyter Notebook containing all the code for data processing, EDA, model training, and evaluation.

spam.csv: The dataset used for this project.

Steps Involved
The project follows a standard machine learning pipeline:

1. Data Cleaning
Column Dropping: Removed the last three unnamed columns (Unnamed: 2, Unnamed: 3, Unnamed: 4) as they contain mostly null values and are not relevant for classification.

Column Renaming: Renamed v1 to target and v2 to text for better readability.

Label Encoding: Converted the target column ('ham' and 'spam') into numerical representations (0 and 1) using LabelEncoder.

Missing Values: Checked for and confirmed no missing values in the relevant columns.

Duplicate Removal: Identified and removed duplicate SMS entries to ensure data integrity and prevent model bias.

2. Exploratory Data Analysis (EDA)
Target Distribution: Analyzed the distribution of 'ham' and 'spam' messages.

Observation: The dataset is imbalanced, with a significantly higher number of 'ham' messages compared to 'spam' messages.

Counts:

ham: 4516

spam: 653

Feature Engineering: Added new features to the dataset:

num-characters: Number of characters in each SMS.

num-words: Number of words in each SMS (using nltk.word_tokenize).

num-sentences: Number of sentences in each SMS (using nltk.sent_tokenize).

Statistical Summary of New Features:

Overall:
| | num-characters | num-words | num-sentences |
|---|---|---|---|
| count | 5169.00 | 5169.00 | 5169.00 |
| mean | 78.98 | 18.46 | 1.97 |
| std | 58.24 | 13.32 | 1.45 |
| min | 2.00 | 1.00 | 1.00 |
| 25% | 36.00 | 9.00 | 1.00 |
| 50% | 60.00 | 15.00 | 1.00 |
| 75% | 117.00 | 26.00 | 2.00 |
| max | 910.00 | 220.00 | 38.00 |

Ham Messages:
| | num-characters | num-words | num-sentences |
|---|---|---|---|
| count | 4516.00 | 4516.00 | 4516.00 |
| mean | 70.46 | 17.12 | 1.82 |
| std | 56.36 | 13.49 | 1.38 |
| min | 2.00 | 1.00 | 1.00 |
| 25% | 34.00 | 8.00 | 1.00 |
| 50% | 52.00 | 13.00 | 1.00 |
| 75% | 90.00 | 22.00 | 2.00 |
| max | 910.00 | 220.00 | 38.00 |

Spam Messages:
| | num-characters | num-words | num-sentences |
|---|---|---|---|
| count | 653.00 | 653.00 | 653.00 |
| mean | 137.89 | 27.67 | 2.97 |
| std | 30.14 | 7.01 | 1.49 |
| min | 13.00 | 2.00 | 1.00 |
| 25% | 132.00 | 25.00 | 2.00 |
| 50% | 149.00 | 29.00 | 3.00 |
| 75% | 157.00 | 32.00 | 4.00 |
| max | 224.00 | 46.00 | 9.00 |

3. Text Preprocessing
This step involves transforming raw text data into a format suitable for machine learning models. Common techniques include:

Lowercasing

Tokenization

Removing special characters

Removing stop words and punctuation

Stemming/Lemmatization

4. Model Building
Various machine learning models will be explored and trained for classification. Given the nature of text classification, models like Naive Bayes, Support Vector Machines, Logistic Regression, and potentially deep learning models (e.g., LSTMs) are suitable candidates.

5. Evaluation
The performance of the trained models will be evaluated using metrics such as:

Accuracy Score

Precision Score

Recall Score

F1-Score

Confusion Matrix

6. Improvement
Based on the evaluation results, the models will be fine-tuned and improved. This may involve:

Hyperparameter tuning

Exploring different text vectorization techniques (e.g., TF-IDF, Word2Vec)

Addressing class imbalance (e.g., using SMOTE)

Ensemble methods

7. Website (Future Work)
The plan is to develop a user-friendly web interface where users can input an SMS message and get an instant prediction (ham or spam).

8. Deployment (Future Work)
The final model will be deployed to make it accessible for real-time predictions.

Requirements
To run this project, you will need:

Python 3.x

Jupyter Notebook or Jupyter Lab

Libraries:

numpy

pandas

chardet

nltk

scikit-learn

matplotlib

seaborn (likely used for visualizations in EDA)

You can install the necessary libraries using pip:

pip install numpy pandas chardet nltk scikit-learn matplotlib seaborn

For NLTK data:

import nltk
nltk.download('punkt')
# You might need to download other NLTK data like stopwords if used in preprocessing
# nltk.download('stopwords')

Usage
Clone the repository:

git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier

Place the dataset: Ensure spam.csv is in the root directory of the cloned repository.

Open the Jupyter Notebook:

jupyter notebook sms_spam_classifier.ipynb

Run all cells: Execute the cells in the notebook sequentially to perform data cleaning, EDA, model training, and evaluation.

Results
The sms_spam_classifier.ipynb notebook will show the performance of the models, including accuracy, precision, and confusion matrices. For instance, after improving the model with Multinomial Naive Bayes (MNB) and TF-IDF:

0.9709864603481625
[[896   0]
 [ 30 108]]
0.9782608695652174

This indicates an accuracy of approximately 97.10% and a precision of 97.83% for the MNB model.
