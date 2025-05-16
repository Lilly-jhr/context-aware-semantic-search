
## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python:** Version 3.10 or higher is recommended.
2.  **Pip:** Python package installer.
3.  **Git:** For cloning the repository.
4.  **Ollama:** Install from [https://ollama.ai/](https://ollama.ai/).
5.  **An Ollama LLM Model:** Pull a model that Ollama will serve. For example:
    ```bash
    ollama pull mistral
    ```
    (You can change the model in the `.env` file, e.g., to `gemma:7b` or `llama2`, ensure you pull it first.)
6.  **(Windows Only - Conditional):** If you encounter issues installing `chromadb` (specifically its `hnswlib` dependency), you may need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++" workload during installation). Recent versions of `chromadb` often provide pre-compiled wheels that might not require this.

## ⚙️ Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd context-aware-semantic-search 
    # Replace <YOUR_REPOSITORY_URL> with the actual URL
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy Language Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set Up Environment Variables:**
    *   Copy the `.env.example` file to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and review/modify the variables if needed:
        *   `EMBEDDING_MODEL_NAME`: Defaults to a good general-purpose model.
        *   `COLLECTION_NAME`: Name for the ChromaDB collection.
        *   `OLLAMA_BASE_URL`: Default is `http://localhost:11434`.
        *   `OLLAMA_MODEL`: **Ensure this matches a model you have pulled with Ollama** (e.g., `mistral`, `gemma:7b`).

## ▶️ Running the Application

1.  **Start Ollama Service:** Ensure the Ollama application is running in the background. (How you start it depends on your OS - usually just launching the Ollama app).
2.  **Run the Streamlit App:**
    From the project root directory (`context-aware-semantic-search/`), run:
    ```bash
    streamlit run frontend/app.py --server.fileWatcherType none
    ```
    *The `--server.fileWatcherType none` flag is recommended to avoid potential startup issues related to Streamlit's file watcher and PyTorch on some systems. If you omit it and encounter errors on startup mentioning `torch.classes` or `asyncio event loop`, add the flag.*

3.  **Open in Browser:** Open the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## 📖 Usage Walkthrough

1.  **Sidebar (⚙️ Setup & Manage KB):**
    *   **Upload Documents:** Use the file uploader to select one or more PDF or TXT files for your knowledge base.
    *   **Processing Parameters:** Adjust chunk size and overlap if desired.
    *   **Process Files & Prepare Chat:** Click this button. The application will:
        *   Load and chunk the uploaded documents.
        *   Generate embeddings for these chunks.
        *   Store them in the local ChromaDB vector store.
        *   Initialize the LLM and Conversational AI chain.
    *   Monitor the **System Status** indicators in the sidebar to see when components are ready.

2.  **Main Chat Area (💬 Chat with Selected Knowledge):**
    *   Once the "System Status" in the sidebar indicates the Knowledge Base and Conversational AI are ready, this area will become active.
    *   Type your questions about the uploaded documents into the chat input at the bottom.
    *   The AI will respond, using the documents as context.
    *   For AI responses based on your documents, you can expand **"View Sources & Entities"** to see:
        *   The specific document chunks used by the AI.
        *   A preview of the chunk's content.
        *   Key Named Entities (People, Organizations, etc.) extracted from that chunk.

3.  **Reset:** The "Reset Application State" button in the sidebar will clear the current session, chat history, and attempt to delete the persisted vector store from disk, allowing you to start fresh.

## 🔮 Future Enhancements (Potential To-Do)

*   **📊 Evaluation UI:** Allow users to rate the quality/relevance of answers.
*   **🌌 Embedding Space Visualization:** Implement UMAP/t-SNE + Plotly to visualize document chunk embeddings. 
*   More advanced NLP enrichments (e.g., query expansion, summarization of multiple retrieved chunks).
*   Support for more document types.
*   Option to select from multiple existing knowledge bases.
