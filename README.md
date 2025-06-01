# Complete RAG System 🤖

A comprehensive Retrieval-Augmented Generation (RAG) system built with Mistral-7B, ChromaDB, and Streamlit. This system allows you to upload PDF documents, process them into a vector database, and query them using natural language with an intelligent AI assistant.

## 🌟 Features

- **PDF Document Processing**: Extract and chunk text from PDF files
- **Vector Database Storage**: Store document embeddings in ChromaDB for efficient retrieval
- **Mistral-7B Integration**: Leverage Mistral's powerful language model for answer generation
- **Interactive Chat Interface**: User-friendly Streamlit web interface
- **Analytics Dashboard**: Monitor system performance and query statistics
- **Configurable Settings**: Adjust model parameters, chunk sizes, and retrieval settings
- **4-bit Quantization Support**: Optimize memory usage for large models

## 🏗️ Architecture

The system consists of three main phases:

1. **Phase 1: PDF Processing & Vector Database**
   - PDF text extraction using PyPDF2 and pdfplumber
   - Text chunking with configurable overlap
   - Sentence embedding generation
   - ChromaDB storage with metadata

2. **Phase 2: LLM Chain with Mistral**
   - Document retrieval based on semantic similarity
   - Context creation from retrieved documents
   - Answer generation using Mistral-7B-Instruct
   - Confidence scoring and response evaluation

3. **Phase 3: Streamlit Interface**
   - Interactive chat interface
   - Real-time analytics dashboard
   - System configuration controls
   - File upload and processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 16GB+ RAM (8GB minimum with quantization)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/complete-rag-system.git
cd complete-rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
# streamlit run app.py --server.fileWatcherType=none     
```

## 📦 Dependencies

### Core ML/AI Libraries
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
bitsandbytes>=0.39.0
```

### PDF Processing
```
PyPDF2>=3.0.0
pdfplumber>=0.9.0
```

### Web Interface
```
streamlit>=1.25.0
plotly>=5.15.0
pandas>=1.5.0
```

### Utilities
```
python-dotenv>=1.0.0
pathlib
uuid
logging
re
dataclasses
typing
datetime
```

## 🔧 Configuration

### System Settings

The application provides several configuration options:

#### Database Settings
- **ChromaDB Path**: Location for the vector database storage
- **Embedding Model**: Choice of sentence transformer models
- **Chunk Size**: Size of text chunks (256-1024 tokens)
- **Chunk Overlap**: Overlap between consecutive chunks

#### LLM Settings
- **Model Selection**: Mistral-7B or Mixtral-8x7B variants
- **Quantization**: Enable 4-bit quantization for memory efficiency
- **Temperature**: Control response creativity (0.1-1.0)
- **Max Length**: Maximum response length
- **Top-K Retrieval**: Number of documents to retrieve per query

### Environment Variables

Create a `.env` file for additional configuration:

```env
# Model configurations
DEFAULT_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database settings
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=documents

# System settings
CUDA_VISIBLE_DEVICES=0
MAX_WORKERS=4
```

## 💻 Usage

### 1. Initialize the System

1. Launch the Streamlit application
2. Configure settings in the sidebar
3. Click "🚀 Initialize RAG System"
4. Wait for the model to load (first time may take several minutes)

### 2. Upload Documents

1. Use the file uploader in the sidebar
2. Select one or more PDF files
3. Click "Process Files" to extract and store content
4. Monitor processing progress

### 3. Query Your Documents

1. Use the chat interface to ask questions
2. View retrieved sources and confidence scores
3. Explore analytics in the dashboard tab

### Example Queries

- "What are the main findings in the research papers?"
- "Summarize the methodology used in the studies"
- "Who are the key authors mentioned?"
- "What conclusions can be drawn from the data?"

## 📊 Analytics Dashboard

The system provides comprehensive analytics:

- **System Metrics**: Document count, query statistics
- **Performance Monitoring**: Response times, confidence scores
- **Query Analysis**: Confidence distribution, historical trends
- **Document Overview**: Processing statistics per file

## 🛠️ Advanced Usage

### Custom Model Integration

To use different models, modify the model configuration:

```python
# In the sidebar configuration
model_options = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "your-custom-model-name"
]
```

### Batch Processing

For large document collections, consider batch processing:

```python
# Example batch processing script
rag_pipeline = RAGPipeline()
pdf_files = Path("./documents").glob("*.pdf")

for pdf_file in pdf_files:
    rag_pipeline.process_pdf(pdf_file)
    print(f"Processed: {pdf_file.name}")
```

### API Integration

The core classes can be used independently for API development:

```python
from rag_system import MistralRAGChain

# Initialize chain
rag_chain = MistralRAGChain()

# Process query
response = rag_chain.query("Your question here")
print(response.answer)
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Enable 4-bit quantization
   - Reduce max_length parameter
   - Use smaller batch sizes

2. **Model Loading Errors**
   - Ensure sufficient disk space (>20GB)
   - Check internet connection for model downloads
   - Verify CUDA compatibility

3. **PDF Processing Failures**
   - Ensure PDFs are not password-protected
   - Check file size limits
   - Verify PDF format compatibility

### Performance Optimization

- Use quantization for memory efficiency
- Adjust chunk sizes based on document types
- Monitor system resources during processing
- Consider using smaller embedding models for faster processing

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Development Setup

```bash
# Create development environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black rag_system/
flake8 rag_system/
```


## 🙏 Acknowledgments

- **Mistral AI** for the powerful language models
- **Sentence Transformers** for embedding capabilities
- **ChromaDB** for vector database functionality
- **Streamlit** for the web interface framework
- **Hugging Face** for model hosting and transformers library

## 🗺️ Roadmap

- [ ] Support for additional document formats (DOCX, TXT, etc.)
- [ ] Multi-language document processing
- [ ] Advanced retrieval strategies (hybrid search)
- [ ] Integration with cloud storage services
- [ ] Fine-tuning capabilities for domain-specific models
- [ ] REST API development
- [ ] Docker containerization
- [ ] Kubernetes deployment configurations

---

**Built with ❤️ for the AI community**
