import pandas as pd

# Load datasets
df1 = pd.read_csv("fake_job_postings.csv")  # Fake job postings
df2 = pd.read_csv("postings.csv")    # US real job postings
df3 = pd.read_csv("pakistan_job_postings.csv")  # Pakistan real job postings
# Load first 5000 rows
df = pd.read_csv("postings.csv", nrows=5000)
for df in [df1, df2, df3]:
    df.columns = df.columns.str.strip().str.lower()

# Ensure all datasets have the required columns
columns_to_keep = ["title", "description", "company_profile", "requirements", "employment_type"]
for df in [df1, df2, df3]:
    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = ""  # Add missing columns with empty values
# Select relevant columns
columns_to_keep = ["title", "description", "company_profile", "requirements", "employment_type"]

df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]
df3 = df3[columns_to_keep]
df1 = df1.copy()
df2 = df2.copy()
df3 = df3.copy()
# Add fraudulent labels (1 = Fake, 0 = Real)
df1.loc[:, "fraudulent"] = 1
df2.loc[:, "fraudulent"] = 0
df3.loc[:, "fraudulent"] = 0

# Merge datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Combine text columns into one for analysis
df["job_content"] = df["title"] + " " + df["description"] + " " + df["company_profile"] + " " + df["requirements"]

# Save the final dataset
df.to_csv("merged_job_postings.csv", index=False)

print("Data merged and saved!")
print(df.head())  # View first 5 rows
print(df.info())  # Check column data types
df.fillna("", inplace=True)  # Replace NaN values with empty strings
print("Missing values handled!")
import matplotlib.pyplot as plt

# Count of fraudulent vs. non-fraudulent jobs
df["fraudulent"].value_counts().plot(kind="bar", color=["green", "red"])
plt.xticks(ticks=[0, 1], labels=["Real Jobs (0)", "Fake Jobs (1)"])
plt.xlabel("Job Type")
plt.ylabel("Count")
plt.title("Distribution of Fake vs. Real Job Postings")
plt.show()
from wordcloud import WordCloud

# Generate word clouds for fake and real job descriptions
fake_jobs = df[df["fraudulent"] == 1]["description"].str.cat(sep=" ")
real_jobs = df[df["fraudulent"] == 0]["description"].str.cat(sep=" ")

# Fake job word cloud
plt.figure(figsize=(10,5))
plt.imshow(WordCloud(width=800, height=400, background_color="red").generate(fake_jobs))
plt.axis("off")
plt.title("Most Common Words in Fake Job Postings")
plt.show()

# Real job word cloud
plt.figure(figsize=(10,5))
plt.imshow(WordCloud(width=800, height=400, background_color="green").generate(real_jobs))
plt.axis("off")
plt.title("Most Common Words in Real Job Postings")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm  # progress bar
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimized batch embedding function
def get_bert_embeddings(text_list, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Generating BERT embeddings"):
            batch_texts = text_list[i:i+batch_size]
            tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**tokens)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embeds)
    return torch.cat(embeddings).numpy()
df = df.sample(300, random_state=42)

# Convert text to BERT embeddings
print("Generating BERT embeddings...")
X = get_bert_embeddings(df["job_content"].tolist())
y = df["fraudulent"].values
print("Embeddings generated!")

# Step 2: Apply SMOTE on BERT embeddings
print("Before SMOTE:", dict(pd.Series(y).value_counts()))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("After SMOTE:", dict(pd.Series(y_resampled).value_counts()))
print("Dataset balanced with SMOTE!")

# Step 3: Train classifier (logistic regression or any other)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Save model
import joblib
joblib.dump(clf, "fraud_detection_model.pkl")
joblib.dump(tokenizer, "bert_tokenizer.pkl")
joblib.dump(model, "bert_embedding_model.pkl")
print("Model and tokenizer saved!")
