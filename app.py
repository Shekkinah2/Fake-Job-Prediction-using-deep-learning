from flask import Flask, request, render_template
import joblib
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load model and BERT components
model = joblib.load('fraud_detection_model.pkl')
tokenizer = joblib.load('bert_tokenizer.pkl')
bert_model = joblib.load('bert_embedding_model.pkl')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Get input from form
        job_title = request.form['job_title']
        description = request.form['description']
        company_profile = request.form['company_profile']
        requirements = request.form['requirements']
        employment_type = request.form['employment_type']

        # Combine into one string
        job_content = f"{job_title} {description} {company_profile} {requirements} {employment_type}"
        embedding = get_bert_embedding(job_content)

        # Predict
        prediction = model.predict(embedding)[0]
        result = "Fake Job" if prediction == 1 else "Real Job"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
