import faiss
import numpy as np
import PyPDF2
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

genai.configure(api_key="AIzaSyAkSTjxxoy7CFK8fGiGyS5AiSVyIkScI-U")


# Load BERT-based Sentence Transformer model
embedder = SentenceTransformer("bert-base-nli-mean-tokens")

# Sample job description
job_descriptions = [
    #  Data Science & AI Roles
    "Data Scientist with expertise in Python, SQL, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-Learn, NLP, and Data Analytics.  experience in exploratory data analysis (EDA) and statistical modeling.",
    "Machine Learning Engineer with skills in supervised and unsupervised learning, reinforcement learning, model deployment using Flask/FastAPI, MLOps, and cloud platforms like AWS, GCP, and Azure.",
    "NLP Engineer with experience in text preprocessing, Named Entity Recognition (NER), transformers (BERT, GPT), and vector embeddings for semantic search.",
    
    #  Software Development Roles
    "Software Engineer proficient in Java, Spring Boot, Microservices, REST APIs, and cloud deployment. Strong in design patterns and scalable architecture.",
    "Full Stack Developer with experience in React.js, Node.js, Express.js, MongoDB, and PostgreSQL. Knowledge of CI/CD and Docker is a plus.",
    "Backend Engineer specializing in Golang, PostgreSQL, Kubernetes, and distributed systems.",
    
    # Business & Data Analytics Roles
    "Business Analyst with expertise in Power BI, Tableau, Excel, and data visualization. Strong background in requirements gathering and stakeholder communication.",
    "Data Analyst proficient in SQL, Python (Pandas, NumPy), data cleaning, dashboarding (Power BI, Tableau), and reporting.",
    
    # Cloud & DevOps Roles
    "Cloud Engineer skilled in AWS (EC2, S3, Lambda), Terraform, Kubernetes, and Docker.",
    "DevOps Engineer with experience in CI/CD pipelines, GitHub Actions, Jenkins, Kubernetes, and cloud automation.",
    
    # Cybersecurity & Networking Roles
    "Cybersecurity Analyst with knowledge of network security, penetration testing, SIEM tools, and ethical hacking.",
    "Network Engineer experienced in Cisco networking, firewalls, VPNs, and SD-WAN."
]


# Convert job descriptions to BERT embeddings
job_embeddings = np.array([embedder.encode(desc) for desc in job_descriptions], dtype="float32")


dimension = job_embeddings.shape[1]

# Change FAISS index from L2 to Cosine Similarity
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (for Cosine Similarity)

index.add(job_embeddings)  # Add job descriptions


def extract_text_from_pdf(pdf_file):
    """Extracts clean text from uploaded PDF resume"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted.strip() + "\n"
    return text.lower()  # Convert to lowercase for better matching

def find_best_match(resume_text):
    """Finds the most relevant job description for the given resume"""
    resume_embedding = np.array([embedder.encode(resume_text)], dtype="float32")
    _, best_match_idx = index.search(resume_embedding, 1)  # Retrieve 1 best match
    return job_descriptions[best_match_idx[0][0]]




def generate_feedback(resume_text, job_description):
    """Uses Gemini to analyze resume & suggest improvements"""
    prompt = f"""
    Job Description: {job_description}
    Candidate's Resume: {resume_text}
    
    Analyze the resume based on the job description and provide:
    - A match percentage (0-100%).
    - Key skills missing in the resume.
    - Suggested improvements to increase the match.
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    
    return response.text

st.title("📝 Resume & Job Matching using RAG (Gemini)")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Find the best job match
    best_job = find_best_match(resume_text)
    
    # Generate AI feedback
    feedback = generate_feedback(resume_text, best_job)

    # Display results
    st.subheader("🔍 Best Matched Job Description:")
    st.write(best_job)

    st.subheader("📊 AI Feedback & Suggestions:")
    st.write(feedback)


