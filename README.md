Synopsis Scoring Application
Overview
This Streamlit-based web application evaluates the quality of a user-submitted synopsis based on an uploaded article. It scores the synopsis out of 100 based on content coverage, clarity, and coherence, provides qualitative feedback, and visualizes the score breakdown using a bar chart. The app ensures privacy by anonymizing sensitive data and uses local models to avoid external API calls.
Setup Instructions

Install Python:

Ensure Python 3.8+ is installed.


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run the Application:
streamlit run app.py


Access the App:

Open http://localhost:8501 in your browser.
Enter the password catalyst2025.



Usage

Upload an article (.txt or .pdf) and a synopsis (.txt).
View the total score, score breakdown (Content Coverage, Clarity, Coherence), a bar chart, and qualitative feedback.
Read the privacy note explaining anonymization and data handling.

Scoring Methodology

Content Coverage (50 points): Cosine similarity between sentence embeddings of the article and synopsis using all-MiniLM-L6-v2.
Clarity (25 points): Based on average sentence length, with shorter sentences (~15 words) scoring higher.
Coherence (25 points): Counts transition words (e.g., "however", "therefore") to assess logical flow.
Total Score: Sum of the above, scaled to 100.

Privacy Protection Strategy

Anonymization: Names, dates, and phone numbers are replaced with placeholders (e.g., [NAME], [DATE], [PHONE]) using regex.
No Data Storage: Input texts are processed in memory and not stored post-evaluation.
Local Models: Uses all-MiniLM-L6-v2 for embeddings and distilgpt2 for feedback, avoiding external API calls.
Access Control: Password-protected with SHA-256 hashing.

Dependencies

Python 3.8+
Libraries: Listed in requirements.txt

Project Structure

app.py: Main application code.
requirements.txt: List of dependencies.
README.md: This file.
writeup.md: Scoring and privacy details.

Deployment

For cloud deployment, use platforms like Streamlit Cloud or Heroku.
Push the code to a GitHub repository and deploy using the platform's instructions.
Share the deployed link and GitHub repository for testing.

Notes

Ensure ~4GB RAM for model loading.
Contact the Catallyst team for clarifications.

