# 📬 Spam Detector  

A **machine learning-based spam detection system** that classifies incoming emails as either **"ham"** (not spam) or **"spam"** with high accuracy. The project follows a complete **data science workflow**—from **data preprocessing** to **model training** and **evaluation**.  

---

## 📂 Dataset  

**File:** `spam_ham_dataset.csv`  
This dataset contains **5,000+ emails**, each labeled as either ham or spam.  

| Column      | Description |
|-------------|-------------|
| `text`      | Raw content of the email |
| `label`     | Target label (`ham` or `spam`) |
| `label_num` | Numerical label (`0` = ham, `1` = spam) |

---

## 🛠 Methodology  

### 📝 Preprocessing  
Before training the models, we clean and prepare the email text:  
- **Lowercase Conversion** – Ensures consistency.  
- **Punctuation Removal** – Removes unnecessary symbols.  
- **Stopword Removal** – Removes common words (e.g., *the*, *is*, *a*) that don't help classification.  

---

### 🔬 Feature Extraction  
We use **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into a numerical format:  
- Each row = an email  
- Each column = a word  
- Each value = importance of that word in the email relative to the dataset  

---

### 🧠 Model Training  
We trained and evaluated two models:  

| Model         | Description | Accuracy |
|---------------|-------------|----------|
| **Naive Bayes** | Probabilistic classifier suited for text data | **95.17%** |
| **SVM**        | Finds optimal boundary between classes | **98.84%** ✅ |

**Conclusion:** SVM outperformed Naive Bayes in both accuracy and precision, making it the **preferred model** for this project.  

---

## 📦 Using the Saved Model  

We saved both the **trained SVM model** and the **TF-IDF vectorizer** using `joblib` for easy reuse without retraining.  

**Example Usage:**  
```python
import joblib

# Load the saved model and vectorizer
svm_model = joblib.load('svm_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Preprocess the new email
new_email = "Hello, this is a legitimate email about your project."
cleaned_email = preprocess_text(new_email)  # Define your preprocessing function

# Convert to vector
new_email_vector = tfidf_vectorizer.transform([cleaned_email])

# Predict
prediction = svm_model.predict(new_email_vector)

if prediction[0] == 1:
    print("This email is SPAM! 😠")
else:
    print("This email is HAM. 👍")
```

---

## 👩‍💻 Author  
**Shalini Saurav**  

---

## 🚀 Tech Stack  
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib  
- **Algorithm:** Support Vector Machine (SVM), Naive Bayes  
- **Feature Engineering:** TF-IDF Vectorization  

---

## 📊 Workflow  
1. Data Collection  
2. Data Preprocessing  
3. Feature Extraction (TF-IDF)  
4. Model Training (Naive Bayes, SVM)  
5. Model Evaluation  
6. Model Saving for Deployment  

---

## 📌 Key Highlights  
- High accuracy **(98.84%)** with SVM  
- Reusable **pre-trained model** and **vectorizer**  
- End-to-end **machine learning pipeline**  
