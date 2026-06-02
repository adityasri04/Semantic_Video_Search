
# Contributing to the Project

First off, thank you for taking the time to contribute! 🎉 

This project aims to make video content searchable using natural language by leveraging multimodal AI models, vector databases, and robust CI/CD automation. 
Contributions from developers like you help make this tool more efficient, scalable, and accurate.

---

## 🏗️ Technical Stack Overview
Before diving in, it helps to be familiar with the core components of the engine:
* **Embeddings:** SigLIP (Signal Image-Language Pre-training) for generating dense multimodal vectors.
* **Vector Search:** Qdrant for high-performance indexing and querying.
* **Automation:** Jenkins / GitHub Actions for continuous integration and pipeline orchestration.

---

## 🚀 How to Get Started

### 1) using Docker Compose (Recommended)
#### Fork and Clone the Repository
First, fork the repository to your own GitHub account, then clone it locally:
```
git clone https://github.com/adityasri04/semantic-video-search.git
```
```
cd semantic-video-search
```
#### Run the complete stack including backend + Qdrant.

Start Services
```
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```
```
docker-compose up --build -d
```

Access API Docs
```
http://localhost:8000/docs
```

### Or
#### 2) Fork and Clone the Repository
First, fork the repository to your own GitHub account, then clone it locally:
```
git clone https://github.com/adityasri04/Semantic_Video_Search.git
```
```
cd semantic-video-search
```

#### 2. Set Up Your Environment
We highly recommend using a virtual environment (venv or conda) to keep dependencies isolated.
```Python
python -m venv venv
```

#### Activate it
On Windows:
```CMD
.\\venv\\Scripts\\activate
```
On macOS/Linux:
```Bash
source venv/bin/activate
```

#### Install development dependencies
```
pip install -r requirements.txt
```
#### Run Service
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit 
```
http://127.0.0.1:8000/docs
```



#### 3. Handle Local Metadata & Weights

* **Model Weights:** Since model weights (SigLIP) are large, do not commit them directly to Git. Ensure they are downloaded to the designated .cache/ or models/ directory, which is ignored by .gitignore.
* **Corrupted Metadata:** When testing video parsing features, always run your code against corrupted or edge-case temporal metadata to ensure your changes don't break the ingestion pipeline.

---

## 🔄 Git Workflow & Branching Strategy

We follow a clean branch-and-merge workflow. Please avoid pushing directly to the main branch.

### Branch Naming Conventions
* feat/your-feature-name for new capabilities (e.g., feat/batch-video-processing)
* bug/issue-description for resolving bugs (e.g., bug/corrupted-timestamp-parsing)
* docs/updates for documentation improvements.

### Step-by-Step Workflow
1. Pull latest changes: Always sync your local branch with the upstream repository before starting work.
2. Create your branch: git checkout -b feature/amazing-feature
3. Commit your changes: Write clear, concise commit messages. 
   * Example: git commit -m "Fix temporal metadata indexing for corrupted video frames"
4. Push to your fork: git push origin feature/amazing-feature
5. Open a Pull Request (PR): Submit a PR against the main (or any other) branch of the primary repository.

---

## 🧪 Code Quality & Testing

### Writing Clean Code
* Follow PEP 8 guidelines for Python code.
* Include descriptive docstrings and type hints for new functions, especially around vector manipulation and FAISS index operations.

### Local Verification
Before opening a PR, ensure your changes don't break existing functionality:
* Run unit tests to verify embedding generation and query indexing.
* Verify any CI pipeline configurations locally if you are modifying Jenkinsfiles or GitHub Actions workflows.

---

## 🤝 Community & Code of Conduct

* Be Respectful: Treat all contributors with kindness and respect.
* File Detailed Issues: If you find a bug, open an issue detailing the steps to reproduce it, your environment setup, and the expected vs. actual behavior.

Happy coding! 🎥🔍
