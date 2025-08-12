#Spam Detector 📬✨

This project is a machine learning-based spam detection system. Using a dataset of emails, the goal is to classify new messages as either "ham" (not spam) or "spam" with high accuracy. The project follows a standard data science workflow, from data preprocessing to model training and evaluation.

📂 Dataset

The core of this project is the spam_ham_dataset.csv file. It contains over 5,000 emails, each labeled as either ham or spam. The dataset provides the raw text of each email, which is the key feature we use to train our models.

Column
Description
text
The raw content of the email.
label
The target variable (ham or spam).
label_num

The numerical representation of the label (0 for ham, 1 for spam).

🤖 Methodology

Our approach to building the spam detector involves several key steps to prepare the data and train the models.

📝 Preprocessing

Before feeding the text data to a model, we perform several cleaning steps:

Lowercase Conversion: All text is converted to lowercase to ensure consistency.

Punctuation Removal: All punctuation is stripped away.

Stopword Removal: Common words like "the," "is," and "a" are removed, as they don't typically help in classifying an email as spam or ham.

🔬 Feature Extraction

We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the cleaned text into a numerical format. This technique creates a feature matrix where each row is an email and each column represents a word. The value in each cell indicates the importance of that word within the email relative to the entire dataset.

🧠 Model Training

We trained and evaluated two popular machine learning algorithms for text classification:

Naive Bayes: A probabilistic classifier that works well for text data.

Support Vector Machine (SVM): A powerful algorithm that finds an optimal boundary to separate the classes.

📈 Results & Conclusion

After training both models, we evaluated their performance on a test set of emails they had never seen before.

Model

Accuracy

Naive Bayes

95.17%

SVM

98.84%

The SVM model significantly outperformed the Naive Bayes model. Its high accuracy and precision make it an excellent choice for a robust spam detection system.

📦 Using the Saved Model

To make this project easily reusable, we've saved the trained SVM model and the TF-IDF vectorizer to disk using joblib. This is an efficient alternative to pickle for scikit-learn models.

You can load and use these files to make predictions on new emails without needing to retrain the model.

Here's how you can do it in your Python script:

import joblib

# Load the saved model and vectorizer
svm_model = joblib.load('svm_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Example of a new email to classify
new_email = "Hello, this is a legitimate email about your project."

# Preprocess the new email using the same function
# ... (your preprocess_text function goes here)
cleaned_email = preprocess_text(new_email)

# Transform the cleaned email using the loaded vectorizer
new_email_vector = tfidf_vectorizer.transform([cleaned_email])

# Make a prediction
prediction = svm_model.predict(new_email_vector)

if prediction[0] == 1:
    print("This email is SPAM! 😠")
else:
    print("This email is HAM. 👍")
   👩‍💻 Author :
     Shalini Saurav 
