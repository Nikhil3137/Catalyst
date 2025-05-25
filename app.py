import streamlit as st
import fitz  # PyMuPDF for PDF handling
import re
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import hashlib

# Download required NLTK data
nltk.download('punkt_tab')

# Initialize models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    feedback_generator = pipeline('text-generation', model='distilgpt2')
    return embedder, feedback_generator

embedder, feedback_generator = load_models()

# Password protection
def check_password():
    def password_entered():
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256("catalyst2025".encode()).hexdigest():
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    return True

# Anonymization function
def anonymize_text(text):
    # Replace names with placeholders
    text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[NAME]', text)
    # Replace dates (e.g., 12/25/2023 or 25-12-2023)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    # Replace specific identifiers like phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    return text

# Extract text from PDF or TXT
def extract_text(file, file_type):
    if file_type == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    else:
        text = file.read().decode("utf-8")
    return text

# Score synopsis
def score_synopsis(article_text, synopsis_text):
    # Anonymize texts
    anon_article = anonymize_text(article_text)
    anon_synopsis = anonymize_text(synopsis_text)
    
    # Split into sentences
    article_sentences = nltk.sent_tokenize(anon_article)
    synopsis_sentences = nltk.sent_tokenize(anon_synopsis)
    
    # Calculate embeddings
    article_embeddings = embedder.encode(article_sentences, convert_to_tensor=True)
    synopsis_embeddings = embedder.encode(synopsis_sentences, convert_to_tensor=True) if synopsis_sentences else None
    
    # Content coverage: cosine similarity
    content_coverage = 0
    if synopsis_embeddings is not None and article_embeddings is not None:
        cosine_scores = util.cos_sim(synopsis_embeddings, article_embeddings)
        content_coverage = float(cosine_scores.max(dim=1)[0].mean()) * 50  # Scale to 50
    
    # Clarity: average sentence length (simpler sentences = clearer)
    clarity = 0
    if synopsis_sentences:
        avg_synopsis_len = sum(len(s.split()) for s in synopsis_sentences) / len(synopsis_sentences)
        clarity = min(25, 25 * (15 / max(avg_synopsis_len, 1)))  # Ideal sentence length ~15 words
    
    # Coherence: check transition words
    transition_words = len(re.findall(r'\b(however|therefore|moreover|consequently|nevertheless)\b', anon_synopsis.lower()))
    coherence = min(25, transition_words * 5)  # 5 points per transition word, max 25
    
    total_score = content_coverage + clarity + coherence
    return {
        "total": round(total_score, 2),
        "content_coverage": round(content_coverage, 2),
        "clarity": round(clarity, 2),
        "coherence": round(coherence, 2)
    }

# Generate qualitative feedback
def generate_feedback(scores):
    prompt = f"Synopsis evaluation: Content Coverage {scores['content_coverage']}/50, Clarity {scores['clarity']}/25, Coherence {scores['coherence']}/25. Provide 2-3 lines of constructive feedback."
    feedback = feedback_generator(prompt, max_length=150, num_return_sequences=1, truncation=True, pad_token_id=50256)[0]['generated_text']
    # Clean up feedback
    feedback = feedback.replace(prompt, "").strip()
    feedback_lines = [line for line in feedback.split('\n') if line.strip()][:3]
    return ' '.join(feedback_lines) or "The synopsis needs improvement in covering key points and maintaining clarity."

# Create chart for score breakdown
def create_score_chart(scores):
    chart_data = pd.DataFrame({
        "Metric": ["Content Coverage", "Clarity", "Coherence"],
        "Score": [scores['content_coverage'], scores['clarity'], scores['coherence']],
        "Max Score": [50, 25, 25]
    })
    st.bar_chart(chart_data.set_index("Metric")[["Score", "Max Score"]])

# Main Streamlit app
def main():
    st.title("Synopsis Scoring Application")
    
    if not check_password():
        return
    
    st.write("Upload an article (.txt or .pdf) and its synopsis (.txt) to evaluate the synopsis quality.")
    
    # File upload
    article_file = st.file_uploader("Upload Article (TXT or PDF)", type=["txt", "pdf"])
    synopsis_file = st.file_uploader("Upload Synopsis (TXT)", type=["txt"])
    
    if article_file and synopsis_file:
        try:
            # Extract text
            article_type = "pdf" if article_file.name.endswith(".pdf") else "txt"
            article_text = extract_text(article_file, article_type)
            synopsis_text = extract_text(synopsis_file, "txt")
            
            if not article_text.strip() or not synopsis_text.strip():
                st.error("Uploaded files are empty or invalid. Please upload valid files.")
                return
            
            # Score and feedback
            with st.spinner("Evaluating synopsis..."):
                scores = score_synopsis(article_text, synopsis_text)
                feedback = generate_feedback(scores)
            
            # Display results
            st.subheader("Synopsis Evaluation Results")
            st.metric("Total Score", f"{scores['total']}/100")
            st.subheader("Score Breakdown")
            create_score_chart(scores)
            st.write(f"- Content Coverage: {scores['content_coverage']}/50")
            st.write(f"- Clarity: {scores['clarity']}/25")
            st.write(f"- Coherence: {scores['coherence']}/25")
            st.subheader("Feedback")
            st.write(feedback)
            
            # Privacy note
            st.subheader("Privacy Note")
            st.write("All uploaded content is anonymized by replacing names, dates, and phone numbers with placeholders before processing. No raw data is stored post-processing.")
        
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
    
    else:
        st.info("Please upload both an article and a synopsis to proceed.")

if __name__ == "__main__":
    main()