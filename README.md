# GPT Server - Chat System with Code Analysis

A sophisticated chat system with advanced code analysis, document retrieval, and web search capabilities.

## Features

- **Multi-modal Chat Interface**: Support for text and image inputs
- **Document RAG**: Upload and query documents with vector-based retrieval
- **Code Analysis**: Intelligent code file processing and context-aware responses
- **Web Search Integration**: Real-time web search with content extraction
- **Image Generation**: DALL-E 3 integration for image creation
- **Microsoft Authentication**: Secure OAuth2 authentication
- **Streaming Responses**: Real-time response streaming with SSE
- **Multi-deployment Support**: Multiple AI model deployments

## Directory Structure

```
gpt_server/
├── main.py                 # Application entry point
├── src/                    # Source code
│   ├── __init__.py
│   ├── app_init.py        # Application initialization and FastAPI setup
│   ├── database.py        # Database models and configuration
│   ├── auth.py            # Authentication logic
│   ├── rag_documents.py   # Document RAG functionality
│   ├── query_code.py      # Code analysis functionality
│   ├── query_web.py       # Web search functionality
│   ├── image_service.py   # Image generation service
│   ├── endpoints/         # API endpoints
│   │   ├── __init__.py
│   │   ├── auth.py        # Authentication endpoints
│   │   ├── chat.py        # Chat management endpoints
│   │   ├── files.py       # File upload endpoints
│   │   ├── query.py       # Query processing endpoints
│   │   └── root.py        # Static file serving
│   └── utils/             # Utility modules
│       ├── __init__.py
│       ├── async_helpers.py
│       ├── llm_helpers.py
│       ├── llm_helpers_azure.py
│       ├── rag_utils.py
│       ├── sse.py
│       └── text_processing.py
├── config/                # Configuration files
│   ├── __init__.py
│   ├── config.py          # Environment configuration
│   ├── system_prompt.txt  # System prompt for AI
│   └── avaliable_models.txt # Available AI models
├── deployment/            # Deployment files
│   ├── README.md          # Deployment instructions
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .dockerignore
│   ├── requirements.txt
│   ├── env.sample
│   ├── docker-instructions.md
│   ├── init-db/           # Database initialization
│   └── nginx/             # Nginx configuration
├── data/                  # Application data (created at runtime)
│   ├── chats/             # Chat document storage
│   ├── code/              # Code file storage
│   ├── images/            # Generated images
│   ├── knowledgebase/     # Knowledge base files
│   └── rag/               # RAG index storage
├── static/                # Static web files
└── tools/                 # Development tools
```

## Quick Start

### 1. Environment Setup

Copy the environment template:
```bash
cp deployment/env.sample .env
```

Edit `.env` with your configuration:
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/gpt_server

# Azure OpenAI
AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview

# Microsoft Authentication
MS_CLIENT_ID=your_client_id
MS_CLIENT_SECRET=your_client_secret
MS_TENANT_ID=your_tenant_id
MS_REDIRECT_URI=http://localhost:8000/auth/microsoft/callback

# Security
SECRET_KEY=your_secret_key

# Optional: Web Search
SERPER_API_KEY=your_serper_api_key
```

### 2. Installation

Install dependencies:
```bash
pip install -r deployment/requirements.txt
```

### 3. Database Setup

Initialize the database:
```bash
python -c "import asyncio; from src.database import init_db; asyncio.run(init_db())"
```

### 4. Run the Application

Development mode:
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Docker Deployment

For production deployment with Docker, navigate to the deployment directory:
```bash
cd deployment
docker-compose up -d
```

**Note**: All Docker-related files (Dockerfile, docker-compose.yml, .dockerignore) are located in the `deployment/` directory to keep deployment configuration separate from the application code.

See `deployment/README.md` for detailed deployment instructions.

## API Endpoints

### Authentication
- `GET /auth/microsoft/login` - Initiate Microsoft OAuth login
- `GET /auth/microsoft/callback` - OAuth callback handler
- `GET /user/` - Get current user information

### Chat Management
- `GET /chats/` - Get user's recent chats
- `POST /new_chat/` - Create new chat
- `GET /chat_data/{chat_id}` - Get chat messages and files

### File Operations
- `POST /upload_files/{chat_id}` - Upload documents or code files
- `POST /index_files/{chat_id}` - Index uploaded files for search

### Query Processing
- `GET /query/` - Process general queries with optional web search
- `GET /query_code/` - Process code-specific queries
- `POST /query_image/{chat_id}` - Generate images with DALL-E 3
- `POST /start_query_session/{chat_id}` - Start session for complex queries

### Utility
- `GET /health` - Health check with database status
- `GET /query/available_models` - Get available AI models

## Development

### Project Structure

The application follows a modular architecture:

- **`src/`**: Main application source code
- **`config/`**: Configuration and settings
- **`deployment/`**: Docker and deployment files
- **`data/`**: Runtime data storage (auto-created)

### Key Components

1. **FastAPI Application** (`src/app_init.py`): Core application setup
2. **Database Layer** (`src/database.py`): SQLAlchemy async models
3. **Authentication** (`src/auth.py`): Microsoft OAuth2 integration
4. **RAG System** (`src/rag_documents.py`): Document processing and retrieval
5. **Code Analysis** (`src/query_code.py`): Code file processing
6. **Web Search** (`src/query_web.py`): Real-time web search integration

### Adding New Features

1. Create new modules in `src/` for core functionality
2. Add API endpoints in `src/endpoints/`
3. Add utility functions in `src/utils/`
4. Update configuration in `config/` if needed

## License

This project is licensed under the MIT License. 