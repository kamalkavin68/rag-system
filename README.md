# 🚀 Google Drive RAG System

A powerful **Retrieval-Augmented Generation (RAG)** system that seamlessly integrates with Google Drive, enabling intelligent document querying and AI-powered responses using **Anthropic Claude**, **ChromaDB**, and **Google Embeddings**.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![Anthropic](https://img.shields.io/badge/anthropic-claude--3.5--sonnet-purple.svg)
![Google AI](https://img.shields.io/badge/google--ai-embeddings--001-4285f4.svg)
![ChromaDB](https://img.shields.io/badge/chromadb-vector--store-orange.svg)

## ✨ Features

- 📁 **Google Drive Integration** - Direct document loading from Google Drive folders
- 🔍 **Intelligent Document Search** - Vector-based similarity search using ChromaDB
- 🤖 **AI-Powered Responses** - Context-aware responses using Anthropic Claude 3.5 Sonnet
- 📄 **PDF Processing** - Automated PDF parsing and text extraction
- 🌐 **Web Interface** - User-friendly Streamlit interface
- 💾 **Persistent Storage** - ChromaDB vector store with local persistence
- 🔄 **Real-time Processing** - Live document processing and querying

## 🏗️ System Architecture

![Screenshot](./documentation/rag-architecture.gif)

## Query Processing Diagram

<iframe 
  src="https://viewer.diagrams.net/?tags=%7B%7D&lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&dark=auto#R%3Cmxfile%3E%3Cdiagram%20name%3D%22Page-1%22%20id%3D%22751TUl-cd8_K2EDg3hwz%22%3E7VjbcpswEP0az7QP6XAzwY%2BJ7bTTcSZp0untpaPAGtQKlhFygH59hBE3EzvkMnGayZPR0QrJ5%2BxZSYzMaZh95CQOTtEDNjI0LxuZs5Fh6GNHkz8FkpeIY1sl4HPqqaAGuKT%2FQIFqnL%2BiHiSdQIHIBI27oItRBK7oYIRzTLthS2TdWWPiQw%2B4dAnro9%2BpJwKF6prWdHwC6gdqamesOkJSBSsgCYiHaQsy5yNzyhFF%2BRRmU2AFeRUv5biTLb31wjhEYsiA2e%2FZt7MLdnLlnxlHdBH%2BypLPB2b5lmvCVuoPq8WKvGIAPEmIaiIXAfoYETZv0GOOq8iDYhpNtpqYBWIsQV2Cf0CIXKlLVgIlFIiQqd4lw%2FQoopIyipHCarrWDcHxby2ApO64XGaxtq10KCjBFXdhBwdVWhHug9gRZ9SiyWwHDEHwXI7jwOS6r7vrICrt%2FDquUUY%2BKHHuIZTeE%2BqCpBL4sgL19rZmjSIFe2lABVzGZM1CKn26wT5lbIoM%2BXqs6RFwlm5NeqvHdh24Wg7V5hq4gGy3On021YDaNaps6LZqpy0TVjFBy39V3JMLMH5zyoEx0CnWPp1y%2BCbUYKHsRwq1HnrEOclbATHSSCStN58XQONta9L1tuFsbFwb8VX%2Flnj5UK6gSZn6rzw8i4xeFqlKq33lJEqWyJW0hvZusTh938uxO4put0K%2FxBJsDSnB1nOWYKunSa2FZHLbXvjfC2FbL20vnLyV2LJ0Diixzj73Qrt%2FaoSUUyEger1%2B2Tw7mvq%2B%2FaL3T%2B%2F7MMwd1N%2Fmpyc0jDPQMLp2u7rP4xinp9Q8i8ma99IwFJLXZxnLfHGW0bYewGbgYhhjQsvz12vTwtYHaDF%2BVi32Uq0kXTz%2FUYz%2FYE8mFfBzDUwO7QqYZWqKspW3W%2BfSrpIC4Arc83cmbWAFfOyHpgfdyhxtI%2B2s3beyzfi7bnFj3dkVf99bnGw2n07L8OYDtDm%2FAQ%3D%3D%3C%2Fdiagram%3E%3C%2Fmxfile%3E" 
  width="100%" 
  height="600" 
  frameborder="0" 
  allowfullscreen>
</iframe>



### Architecture Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Source** | Google Drive API | Document storage and access |
| **Document Loader** | LangChain GoogleDriveLoader | PDF parsing and extraction |
| **Text Chunking** | RecursiveCharacterTextSplitter | Document segmentation (1000 chars, 200 overlap) |
| **Embeddings** | Google Generative AI (embedding-001) | Vector representation generation |
| **Vector Database** | ChromaDB | Similarity search and storage |
| **Retrieval** | LangChain VectorStoreRetriever | Context-aware document retrieval |
| **Generation** | Anthropic Claude 3.5 Sonnet | AI response generation |
| **Interface** | Streamlit | Web-based user interface |

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Google Cloud Project with Drive API enabled
- Anthropic API access (via AWS Bedrock)
- Google AI API key

### 1. Clone Repository

```bash
git clone https://github.com/kamalkavin68/rag-system.git
cd rag-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
# Google AI API Key
GOOGLE_API_KEY=your_google_api_key_here

# AWS Credentials for Anthropic Bedrock
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-southeast-2

#  Model configurations
EMBEDDING_MODEL=models/embedding-001
ANTHROPIC_MODEL=arn:aws:bedrock:ap-southeast-2:123367639755:inference-profile/apac.anthropic.claude-3-5-sonnet-20240620-v1:0
```

### 4. Google Drive Authentication

1. Create a Google Cloud Project
2. Enable Google Drive API
3. Create service account credentials
4. Download credentials JSON file
5. Place credentials as `secret/credentials.json`
6. Generate token file (will be created automatically on first run)

```bash
mkdir secret
# Place your credentials.json file in the secret folder
# Place your token.json file in the secret folder
```

### 5. Run Application

```bash
streamlit run main.py
```

### 6. Run Debug Mode

```bash
# Run with debug logging
streamlit run main.py --logger.level=debug
```


Navigate to `http://localhost:8501` in your browser.

## 📊 Performance Optimizations

### Current Optimizations

#### 🔧 **Document Processing**
- **Chunking Strategy**: 1000 character chunks with 200 character overlap for optimal context preservation
- **Batch Processing**: Efficient document loading and processing pipeline
- **Memory Management**: Session state management for persistent data

#### 🚀 **Vector Operations**
- **Similarity Search**: Top-K retrieval (K=4) for relevant context
- **Persistent Storage**: ChromaDB local persistence reduces reprocessing
- **Embedding Caching**: Google's embedding-001 model for consistent vectors

#### 🎯 **Response Generation**
- **Context Optimization**: Focused context injection for relevant responses
- **Token Management**: 5000 max tokens for comprehensive responses
- **Chat History**: Maintained conversation context for better continuity


## 💡 Usage Examples

### Basic Document Query

```python
# Example: Loading HR Policy Documents
folder_id = "1a2b3c4d5e6f7g8h9i0j"  # Your Google Drive folder ID
query = "What is the company's vacation policy?"

# Expected Response:
# "Based on the HR policy document, employees are entitled to 
# 15 days of paid vacation per year, with additional days 
# earned based on tenure..."
```


## 📈 Future Enhancements

- **Multi-format Support**: Word docs, PowerPoint, Excel files
    * Currently only processes PDF files. This would add support for other document formats.
- **Advanced Search**: Hybrid search combining dense and sparse retrieval
    * Combines two search methods for better results (like Dense retrieval and Sparse retrieval)
- **User Authentication**: Multi-user support with access controls
    * Allow multiple users with different access levels and personal chat histories.
- **Real-time Indexing**: Automatic document updates and re-indexing
    * Automatically detect when documents are added/modified in Google Drive and update the vector database.
- **Analytics Dashboard**: Usage statistics and performance metrics
    * Visual dashboard showing system usage, popular queries, response times, and user activity.
- **API Endpoints**: REST API for programmatic access
    * REST API allowing other applications to interact with your RAG system programmatically.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Anthropic](https://anthropic.com/) for Claude AI model
- [Google AI](https://ai.google/) for embedding models
- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage

---



**Built with ❤️ and curiosity for AI innovation**

[⭐ Star this repo]( https://github.com/kamalkavin68/rag-system) | [🐛 Report Bug](https://github.com/kamalkavin68/rag-system/issues) | [💡 Request Feature](https://github.com/kamalkavin68/rag-system/issues)
