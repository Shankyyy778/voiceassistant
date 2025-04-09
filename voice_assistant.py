import os
import time
import threading
import base64
import json
from flask import Flask, render_template, request, jsonify, session
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# Set API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_3qTZgK_EYEdZiqF7Biyyt1FVgEz6CtQbFEmS2jD7pjyRy6F29WWzA9pUpHVdbyT5qh3hxk"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'voice-assistant-secret-key'

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.3)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
index_name = "hari2"

# Load the Pinecone index
try:
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Successfully connected to Pinecone vector store")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    # Fallback to empty retriever
    retriever = None

# Custom prompt template
qa_template = """You are Maya, a friendly voice assistant for Interia, a North India based interior designing company. 
Always be professional, concise, and respond directly to the client's questions or statements.

Context: {context}

History: {chat_history}

Current Question: {question}

Instructions:
1. IMPORTANT: Only introduce yourself in the very first message. Never repeat your introduction or name in subsequent responses.
2. Keep all responses brief - maximum 2-3 sentences.
3. Directly respond to what the client just said without repeating previous questions.
4. When a client mentions a specific need (like kitchen design), immediately focus on that and ask a relevant follow-up question.
5. Progress the conversation forward with each response, don't circle back to previous questions.
6. Use a warm, professional tone like in a sales call.
7. Avoid generic responses that could apply to any input.
8. When discussing projects, briefly ask about budget range, timeline, or requirements.
9. If you don't know an answer, acknowledge it briefly and offer to connect them with an expert.

Your Response:"""

QA_PROMPT = PromptTemplate(
    template=qa_template, 
    input_variables=["context", "question", "chat_history"]
)

# Initialize conversation chain
if retriever:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
else:
    print("Warning: Running without retriever - only basic Q&A will be available")
    qa_chain = None

@app.route('/')
def index():
    return render_template('win_3.html')

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query')
        voice = data.get('voice', 'en')
        chat_history = data.get('history', [])
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        # Format chat history for Langchain
        formatted_history = []
        for msg in chat_history:
            if msg['role'] in ['user', 'assistant']:
                if len(formatted_history) < 4:  # Keep history manageable
                    formatted_history.append((msg['content'] if msg['role'] == 'user' else "", 
                                            msg['content'] if msg['role'] == 'assistant' else ""))
        
        # Generate response using the QA chain if available
        if qa_chain and retriever:
            response = qa_chain({"question": user_query, "chat_history": formatted_history})
            answer = response["answer"]
        else:
            # Fallback to direct LLM call if retriever not available
            answer = llm.invoke(f"Q: {user_query}\nA:")
            answer = answer.content
        
        # Generate speech
        audio_base64 = text_to_speech(answer, voice)
        
        return jsonify({
            "text": answer,
            "audio_data": audio_base64,
            "content_type": "audio/mp3"
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

def text_to_speech(text, lang='en'):
    """Convert text to speech using Google Text-to-Speech"""
    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to a BytesIO object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Convert to base64 for sending over HTTP
        audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
        return audio_base64
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

@app.route('/api/download-report', methods=['POST'])
def download_report():
    # For compatibility with the existing front-end template
    # Return a simple JSON response
    return jsonify({"status": "Report generation not implemented in this version"})

if __name__ == '__main__':
    app.run(debug=True, port=8888) 