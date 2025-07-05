# voiceassistant
voice assistant for interior
### ✅ **README.md for GitHub**

```markdown
# 🎙️ Maya – Voice Assistant for Interia (AI + LangChain + Pinecone + Flask)

Maya is a professional voice assistant designed for **Interia**, a North India-based interior designing company. Powered by **LangChain**, **Groq LLM (LLaMA3)**, **Pinecone vector store**, and **Google Text-to-Speech**, Maya provides fast, context-aware, and professional voice responses for customer queries.

---

## 🚀 Features

- 🤖 Conversational AI using **Groq + LangChain**
- 🔍 Smart Retrieval with **Pinecone Vector Store**
- 🎤 Voice output using **Google Text-to-Speech**
- 📦 Embedding via **Ollama’s mxbai-embed-large**
- 🧠 Memory using **ConversationalRetrievalChain**
- 🌐 Built with **Flask** and REST APIs
- 🛠️ Easily extendable for deployment or UI integration

---

## 🛠️ Tech Stack

- Python, Flask
- LangChain (Groq LLM, Pinecone Retriever, PromptTemplate)
- Google Text-to-Speech (gTTS)
- Pinecone Vector DB
- Ollama Embeddings
- HTML/CSS frontend (template: `win_3.html`)

---

## 📂 Directory Structure

```

project/
│
├── app.py                  # Main Flask app
├── templates/
│   └── win\_3.html          # Frontend UI
├── static/                 # Optional: CSS/JS
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/maya-voice-assistant.git
cd maya-voice-assistant
````

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add API Keys (Groq + Pinecone)

Set your environment variables in `app.py`:

```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"
```

Or use a `.env` file (optional).

---

## ▶️ Run the App

```bash
python app.py
```

Visit: `http://localhost:8888`

---

## 📡 API Endpoint

### POST `/api/query`

**Request:**

```json
{
  "query": "Tell me about kitchen designs",
  "voice": "en",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help you today?"}
  ]
}
```

**Response:**

```json
{
  "text": "We specialize in modular kitchen designs...",
  "audio_data": "BASE64_ENCODED_MP3",
  "content_type": "audio/mp3"
}
```

---

## 🧠 Prompt Template Rules

* Introduce yourself **only in the first message**.
* Be professional and **brief** (2–3 sentences).
* Focus on the **client’s current question** only.
* Drive the conversation forward.
* Don’t repeat previous content or generic lines.

---

## 📌 To Do / Extend

* Add multilingual voice support
* Improve chat history persistence
* Integrate report generation
* Deploy on Render/Heroku

---

## 🧑‍💻 Author

Built by **Shanky** (Harishanker Selvaganapathy)
Specialized in GenAI, NLP, and LangChain-based systems


