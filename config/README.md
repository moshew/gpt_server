# Configuration Guide

## Setting up your .env file

1. Copy the example configuration below into a `.env` file in the root directory:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/gpt_server

# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_api_key_here
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview

# Azure DALL-E Configuration (optional)
DALLE_API_KEY=your_dalle_api_key_here
DALLE_ENDPOINT=https://your-dalle-resource.openai.azure.com/
DALLE_API_VERSION=2024-02-01

# Microsoft Authentication (optional)
MS_CLIENT_ID=your_microsoft_client_id
MS_CLIENT_SECRET=your_microsoft_client_secret
MS_TENANT_ID=your_microsoft_tenant_id
MS_REDIRECT_URI=http://localhost:8000/auth/callback

# JWT Secret Key (required)
SECRET_KEY=your_secret_key_for_jwt_tokens

# Google Serper API Key for Web Search (optional)
SERPER_API_KEY=your_serper_api_key

# Rate Limiting Configuration (optional - defaults shown)
MAX_CONCURRENT_LLM_CALLS=3
MAX_CONCURRENT_EMBEDDING_CALLS=2

# Debug Mode (optional)
DEBUG_MODE=false
```

## Rate Limiting Configuration

### MAX_CONCURRENT_LLM_CALLS
- **Default**: 3
- **Description**: Maximum number of concurrent LLM (chat completion) requests
- **Recommended**: 3-5 for most use cases
- **Higher values**: More parallel processing but risk of 429 errors
- **Lower values**: More conservative, better for limited rate limits

### MAX_CONCURRENT_EMBEDDING_CALLS
- **Default**: 2
- **Description**: Maximum number of concurrent embedding requests
- **Recommended**: 2-3 for most use cases
- **Note**: Embedding calls are typically more expensive than LLM calls

## Monitoring Rate Limits

You can monitor the current rate limiting status by accessing:

```
GET /debug/rate_limits
```

Example response:
```json
{
  "rate_limiting": {
    "llm_calls": {
      "max_concurrent": 3,
      "available_slots": 1,
      "in_use": 2,
      "locked": true
    },
    "embedding_calls": {
      "max_concurrent": 2,
      "available_slots": 0,
      "in_use": 2,
      "locked": true
    }
  },
  "active_chat_requests": 3,
  "active_requests": ["chat_data_21", "chat_data_22", "chat_data_23"]
}
```

## Troubleshooting

### Getting 429 errors frequently?
- Reduce `MAX_CONCURRENT_LLM_CALLS` to 2
- Reduce `MAX_CONCURRENT_EMBEDDING_CALLS` to 1

### Responses seem slow?
- Increase `MAX_CONCURRENT_LLM_CALLS` to 4-5
- Monitor for 429 errors and adjust accordingly

### Many duplicate requests?
- Check the `active_chat_requests` in the debug endpoint
- The system automatically deduplicates requests for the same chat 