import os
import uuid
import time
import json
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import re

# Core dependencies
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# PDF processing
import PyPDF2
import pdfplumber

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PHASE 1: PDF PROCESSING AND VECTOR DATABASE
# =============================================================================

class RAGPipeline:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 db_path: str = "./chroma_db"):
        """
        Initialize RAG Pipeline for document processing
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_path = db_path

        # Initialize sentence transformer
        self.embedding_model = SentenceTransformer(model_name)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def extract_text_from_pdf(self, pdf_file) -> Dict[str, Any]:
        """Extract text from PDF file (handles both file paths and uploaded files)"""
        text_content = []

        # Handle both file paths and Streamlit uploaded files
        if hasattr(pdf_file, 'read'):
            # Streamlit uploaded file
            filename = pdf_file.name
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer
        else:
            # File path
            filename = Path(pdf_file).name
            with open(pdf_file, 'rb') as f:
                pdf_bytes = f.read()

        metadata = {
            "filename": filename,
            "total_pages": 0,
            "extraction_date": datetime.now().isoformat()
        }

        try:
            # Using PyPDF2 for uploaded files
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            metadata["total_pages"] = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    page_text = self._clean_text(page_text)
                    text_content.append({
                        "text": page_text,
                        "page_number": page_num
                    })

        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise Exception(f"PDF extraction failed: {e}")

        return {
            "content": text_content,
            "metadata": metadata
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-"\']', ' ', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def create_chunks(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from document text"""
        chunks = []
        content = document_data["content"]
        base_metadata = document_data["metadata"]

        for page_data in content:
            page_text = page_data["text"]
            page_num = page_data["page_number"]

            if len(page_text) <= self.chunk_size:
                chunks.append({
                    "text": page_text,
                    "metadata": {
                        **base_metadata,
                        "page_number": page_num,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_size": len(page_text)
                    }
                })
            else:
                start = 0
                chunk_num = 1

                while start < len(page_text):
                    end = start + self.chunk_size
                    chunk_text = page_text[start:end]

                    if end < len(page_text):
                        last_period = chunk_text.rfind('.')
                        last_space = chunk_text.rfind(' ')
                        break_point = max(last_period, last_space)

                        if break_point > start + self.chunk_size * 0.7:
                            chunk_text = chunk_text[:break_point + 1]
                            end = start + len(chunk_text)

                    chunks.append({
                        "text": chunk_text.strip(),
                        "metadata": {
                            **base_metadata,
                            "page_number": page_num,
                            "chunk_id": str(uuid.uuid4()),
                            "chunk_number": chunk_num,
                            "chunk_size": len(chunk_text),
                            "start_pos": start,
                            "end_pos": end
                        }
                    })

                    start = end - self.chunk_overlap
                    chunk_num += 1

        return chunks

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=32, convert_to_numpy=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def store_in_chromadb(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """Store chunks and embeddings in ChromaDB"""
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks_with_embeddings:
            ids.append(chunk["metadata"]["chunk_id"])
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])

            flat_metadata = {}
            for key, value in chunk["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    flat_metadata[key] = value
                else:
                    flat_metadata[key] = str(value)

            metadatas.append(flat_metadata)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def process_pdf(self, pdf_file) -> int:
        """Complete pipeline: PDF -> Chunks -> Embeddings -> ChromaDB"""
        document_data = self.extract_text_from_pdf(pdf_file)
        chunks = self.create_chunks(document_data)
        chunks_with_embeddings = self.generate_embeddings(chunks)
        self.store_in_chromadb(chunks_with_embeddings)
        return len(chunks_with_embeddings)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "database_path": self.db_path
        }

# =============================================================================
# PHASE 2: LLM CHAIN WITH MISTRAL
# =============================================================================

@dataclass
class RAGResponse:
    """Structure for RAG response"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence_score: float
    retrieval_time: float
    generation_time: float

class MistralRAGChain:
    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "documents",
               
                 use_quantization: bool = True,
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_k: int = 3):
        """Initialize Mistral RAG Chain"""
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        truncation=True, 
        self.top_k = top_k

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)

        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            # Create collection if it doesn't exist
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # Initialize Mistral model
        self._setup_mistral_model(use_quantization)
        self._create_generation_pipeline()

    def _setup_mistral_model(self, use_quantization: bool):
        """Setup Mistral model with optional quantization"""
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
            truncation=True,  # Add this line
            max_length=self.max_length  # Add this line
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    def _create_generation_pipeline(self):
        """Create text generation pipeline"""
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            truncation=True,  # Add this line
            max_length=self.max_length,  # Add this line
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def retrieve_documents(self, query: str, n_results: int = None) -> tuple:
        """Retrieve relevant documents from ChromaDB"""
        start_time = time.time()

        if n_results is None:
            n_results = self.top_k

        query_embedding = self.embedding_model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,
                    "rank": i + 1
                })

        retrieval_time = time.time() - start_time
        return retrieved_docs, retrieval_time

    def create_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]
            content = doc["content"]

            source_info = f"Source {i}"
            if "filename" in metadata:
                source_info += f" ({metadata['filename']}"
                if "page_number" in metadata:
                    source_info += f", page {metadata['page_number']}"
                source_info += ")"

            context_parts.append(f"{source_info}:\n{content}\n")

        return "\n".join(context_parts)

    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Mistral model"""
        prompt_template = """<s>[INST] You are a helpful assistant that answers questions based on the provided context. Use the context to answer the question accurately and concisely. If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above. [/INST]"""

        return prompt_template.format(context=context, query=query)

    def generate_answer(self, prompt: str) -> tuple:
        """Generate answer using Mistral model"""
        start_time = time.time()

        try:
            response = self.generator(
                prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                truncation=True,  # Add this line
                padding=True,     # Add this for consistency
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            answer = response[0]["generated_text"].strip()
            generation_time = time.time() - start_time

            return answer, generation_time

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}", time.time() - start_time

    def calculate_confidence(self, query: str, answer: str, sources: List[Dict]) -> float:
        """Calculate confidence score"""
        if not sources:
            return 0.0

        avg_similarity = sum(doc["similarity_score"] for doc in sources[:3]) / min(3, len(sources))
        answer_length_score = min(len(answer.split()) / 50, 1.0)

        uncertainty_phrases = [
            "i don't know", "cannot be found", "not mentioned",
            "unclear", "insufficient information"
        ]
        uncertainty_penalty = 0.3 if any(phrase in answer.lower() for phrase in uncertainty_phrases) else 0

        confidence = (avg_similarity * 0.7 + answer_length_score * 0.3) - uncertainty_penalty
        return max(0.0, min(1.0, confidence))

    def query(self, question: str, n_results: int = None) -> RAGResponse:
        """Main query method - complete RAG pipeline"""
        retrieved_docs, retrieval_time = self.retrieve_documents(question, n_results)

        if not retrieved_docs:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base.",
                sources=[],
                query=question,
                confidence_score=0.0,
                retrieval_time=0.0,
                generation_time=0.0
            )

        context = self.create_context(retrieved_docs)
        prompt = self.create_prompt(question, context)
        answer, generation_time = self.generate_answer(prompt)
        confidence = self.calculate_confidence(question, answer, retrieved_docs)

        return RAGResponse(
            answer=answer,
            sources=retrieved_docs,
            query=question,
            confidence_score=confidence,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        collection_stats = {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }

        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "CPU",
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_k_retrieval": self.top_k,
            "collection_stats": collection_stats
        }

# =============================================================================
# PHASE 3: STREAMLIT INTERFACE
# =============================================================================

# Streamlit page configuration
st.set_page_config(
    page_title="ChromaRAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
    }

    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }

    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }

    .source-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .stAlert > div {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class CompleteRAGApp:
    def __init__(self):
        """Initialize the Complete RAG Application"""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None

        if 'rag_chain' not in st.session_state:
            st.session_state.rag_chain = None

        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'uploaded_files_info' not in st.session_state: # Renamed from uploaded_files for clarity
            st.session_state.uploaded_files_info = []

    def initialize_rag_pipeline(self, embedding_model: str, chunk_size: int, chunk_overlap: int, db_path: str):
        """Initialize the RAG pipeline for document processing"""
        try:
            st.session_state.rag_pipeline = RAGPipeline(
                model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_path=db_path
            )
            return True
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {str(e)}")
            return False

    def initialize_rag_chain(self, model_name: str, db_path: str, use_quantization: bool,
                             temperature: float, max_length: int, top_k: int):
        """Initialize the RAG chain with LLM"""
        try:
            with st.spinner("Loading Mistral model... This may take several minutes."):
                st.session_state.rag_chain = MistralRAGChain(
                    model_name=model_name,
                    chroma_db_path=db_path,
                    use_quantization=use_quantization,
                    temperature=temperature,
                    max_length=max_length,
                    top_k=top_k
                )
                st.session_state.system_stats = st.session_state.rag_chain.get_system_info()
                return True
        except Exception as e:
            st.error(f"Error loading RAG system: {str(e)}")
            return False

    def render_sidebar(self):
        """Render the sidebar with configuration and controls"""
        st.sidebar.markdown("## 🔧 System Configuration")

        # Database Configuration
        with st.sidebar.expander("💾 Database Settings", expanded=True):
            db_path = st.text_input("ChromaDB Path", value="./chroma_db")
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"]
            )
            chunk_size = st.slider("Chunk Size", 256, 1024, 512)
            chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)

        # Model Configuration
        with st.sidebar.expander("🤖 LLM Settings"):
            model_name = st.selectbox(
                "Mistral Model",
                ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
            )
            use_quantization = st.checkbox("Use 4-bit Quantization", value=True)
            temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
            max_length = st.slider("Max Response Length", 256, 2048, 1024, 128)
            top_k = st.slider("Documents to Retrieve", 1, 10, 5)

        # Initialize Systems
        if st.sidebar.button("🚀 Initialize RAG System", type="primary"):
            # Initialize pipeline first
            if self.initialize_rag_pipeline(embedding_model, chunk_size, chunk_overlap, db_path):
                st.success("✅ RAG Pipeline initialized!")

                # Then initialize chain
                if self.initialize_rag_chain(model_name, db_path, use_quantization,
                                             temperature, max_length, top_k):
                    st.success("✅ Complete RAG System loaded!")
                    st.rerun()

        # System Status
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 System Status")

        if st.session_state.rag_chain:
            st.sidebar.markdown("🟢 **Status:** Ready")
            stats = st.session_state.system_stats
            if 'collection_stats' in stats:
                st.sidebar.metric("Documents in DB", stats['collection_stats'].get('total_documents', 0))
        else:
            st.sidebar.markdown("🔴 **Status:** Not Loaded")

        # File Upload
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📄 Upload Documents")

        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )

        if uploaded_files and st.sidebar.button("Process Files"):
            self.process_uploaded_files(uploaded_files)

        # Chat Controls
        st.sidebar.markdown("---")
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files"""
        if not st.session_state.rag_pipeline:
            st.error("Please initialize the RAG pipeline first!")
            return

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_chunks = 0
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                document_data = st.session_state.rag_pipeline.extract_text_from_pdf(file)
                chunks = st.session_state.rag_pipeline.create_chunks(document_data)
                chunks_with_embeddings = st.session_state.rag_pipeline.generate_embeddings(chunks)
                st.session_state.rag_pipeline.store_in_chromadb(chunks_with_embeddings)

                total_chunks += len(chunks_with_embeddings)
                st.session_state.uploaded_files_info.append({
                    "filename": document_data["metadata"]["filename"],
                    "total_pages": document_data["metadata"]["total_pages"],
                    "chunks_added": len(chunks_with_embeddings)
                })
                progress_bar.progress((i + 1) / len(uploaded_files))

            # Update system stats
            if st.session_state.rag_chain:
                st.session_state.system_stats = st.session_state.rag_chain.get_system_info()

            st.success(f"✅ Processed {len(uploaded_files)} files ({total_chunks} chunks)")

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")

    def render_chat_message(self, message: Dict[str, Any], is_user: bool = True):
        """Render a single chat message"""
        if is_user:
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])

                # Show sources if available
                if 'sources' in message and message['sources']:
                    with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                        for i, source in enumerate(message['sources'][:3], 1):
                            st.markdown(f"""
                            **Source {i}** (Similarity: {source.get('similarity_score', 0):.2f})

                            📄 *{source.get('metadata', {}).get('filename', 'Unknown')}*
                            (Page {source.get('metadata', {}).get('page_number', 'N/A')})

                            {source.get('content', '')[:200]}...
                            """)

                # Show metrics
                if 'metrics' in message:
                    metrics = message['metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{metrics.get('confidence_score', 0):.2f}")
                    with col2:
                        st.metric("Retrieval", f"{metrics.get('retrieval_time', 0):.2f}s")
                    with col3:
                        st.metric("Generation", f"{metrics.get('generation_time', 0):.2f}s")

    def handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        if not st.session_state.rag_chain:
            st.error("Please initialize the RAG system first!")
            return

        # Add user message
        user_message = {"role": "user", "content": user_input, "timestamp": datetime.now()}
        st.session_state.messages.append(user_message)

        # Generate response
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.rag_chain.query(user_input)

                assistant_message = {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "metrics": {
                        "confidence_score": response.confidence_score,
                        "retrieval_time": response.retrieval_time,
                        "generation_time": response.generation_time
                    },
                    "timestamp": datetime.now()
                }

                st.session_state.messages.append(assistant_message)
                st.session_state.chat_history.append({
                    "query": user_input,
                    "response": response.answer,
                    "confidence": response.confidence_score,
                    "retrieval_time": response.retrieval_time,
                    "generation_time": response.generation_time,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    def render_main_chat(self):
        """Render the main chat interface"""
        st.markdown('<h1 class="main-header">🤖 Complete RAG System</h1>', unsafe_allow_html=True)

        # Display chat messages
        for message in st.session_state.messages:
            self.render_chat_message(message, is_user=(message["role"] == "user"))

        # Chat input
        if user_input := st.chat_input("Ask a question about your documents..."):
            self.handle_user_input(user_input)
            st.rerun()

        # Example questions for new users
        if not st.session_state.messages and st.session_state.rag_chain:
            st.markdown("### 💡 Try these example questions:")
            example_questions = [
                "What are the main topics discussed in the documents?",
                "Can you summarize the key findings?",
                "What methodology was used?",
                "Who are the main authors or contributors?"
            ]

            cols = st.columns(2)
            for i, question in enumerate(example_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"example_{i}"):
                        self.handle_user_input(question)
                        st.rerun()

    def render_analytics(self):
        """Render analytics dashboard"""
        st.header("📊 System Analytics")

        if st.session_state.system_stats:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_docs = st.session_state.system_stats.get('collection_stats', {}).get('total_documents', 0)
                st.metric("Chunks in DB", total_docs) # Changed "Documents" to "Chunks" for accuracy

            with col2:
                total_queries = len(st.session_state.chat_history)
                st.metric("Total Queries", total_queries)

            # Calculate and display average metrics
            if total_queries > 0:
                avg_confidence = sum([q['confidence'] for q in st.session_state.chat_history]) / total_queries
                avg_retrieval_time = sum([q['retrieval_time'] for q in st.session_state.chat_history]) / total_queries
                avg_generation_time = sum([q['generation_time'] for q in st.session_state.chat_history]) / total_queries

                with col3:
                    st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
                with col4:
                    st.metric("Avg. Latency", f"{avg_retrieval_time + avg_generation_time:.2f}s")
            else:
                with col3:
                    st.metric("Avg. Confidence", "N/A")
                with col4:
                    st.metric("Avg. Latency", "N/A")

            st.markdown("---")
            st.subheader("Query Confidence Distribution")
            if st.session_state.chat_history:
                confidence_scores = [q['confidence'] for q in st.session_state.chat_history]
                df_confidence = pd.DataFrame(confidence_scores, columns=['Confidence Score'])
                fig_confidence = px.histogram(df_confidence, x='Confidence Score', nbins=10,
                                              title='Distribution of Query Confidence Scores',
                                              labels={'Confidence Score': 'Confidence Score (0-1)'},
                                              color_discrete_sequence=['#1e3a8a'])
                st.plotly_chart(fig_confidence, use_container_width=True)
            else:
                st.info("No queries yet to display confidence distribution.")

            st.markdown("---")
            st.subheader("Processed Documents Overview")
            if st.session_state.uploaded_files_info:
                df_files = pd.DataFrame(st.session_state.uploaded_files_info)
                fig_files = px.bar(df_files, x='filename', y='total_pages',
                                   title='Total Pages per Processed Document',
                                   labels={'filename': 'Document Name', 'total_pages': 'Total Pages'},
                                   color_discrete_sequence=['#1e3a8a'])
                st.plotly_chart(fig_files, use_container_width=True)

                fig_chunks = px.bar(df_files, x='filename', y='chunks_added',
                                    title='Number of Chunks Added per Document',
                                    labels={'filename': 'Document Name', 'chunks_added': 'Number of Chunks'},
                                    color_discrete_sequence=['#1e3a8a'])
                st.plotly_chart(fig_chunks, use_container_width=True)
            else:
                st.info("No documents processed yet to display document overview.")

        else:
            st.info("Initialize the RAG system to view analytics.")


    def run(self):
        """Main method to run the Streamlit app"""
        self.render_sidebar()

        tab1, tab2 = st.tabs(["💬 Chat Interface", "📈 Analytics Dashboard"])

        with tab1:
            self.render_main_chat()

        with tab2:
            self.render_analytics()

if __name__ == "__main__":
    app = CompleteRAGApp()
    app.run()
