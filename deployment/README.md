# Deployment Guide

This directory contains all the files needed to deploy the GPT Server application using Docker.

## Files Overview

- `Dockerfile` - Production Docker image with Gunicorn + Uvicorn workers
- `Dockerfile.dev` - Development Docker image with direct Uvicorn (for SSE testing)
- `docker-compose.yml` - Production deployment with Nginx reverse proxy
- `docker-compose.dev.yml` - Development deployment without Nginx (for SSE debugging)
- `requirements.txt` - Python dependencies
- `env.sample` - Environment variables template
- `nginx/` - Nginx configuration files
- `init-db/` - Database initialization scripts

## Quick Start

### Production Deployment

```bash
# Copy environment template
cp env.sample ../.env

# Edit .env with your configuration
nano ../.env

# Run production deployment
docker-compose up -d
```

### Development/SSE Testing

If you're experiencing SSE streaming issues, use the development setup without Nginx:

```bash
# Run development deployment (direct uvicorn, no nginx)
docker-compose -f docker-compose.dev.yml up -d
```

This setup:
- Uses direct Uvicorn instead of Gunicorn
- Bypasses Nginx reverse proxy
- Helps identify if buffering issues are caused by Nginx or Gunicorn

## SSE Streaming Fixes

The following optimizations have been implemented for proper SSE streaming in Docker:

### Nginx Configuration (`nginx/nginx.conf`)
- `proxy_buffering off` - Disables response buffering
- `proxy_cache off` - Disables caching
- `proxy_request_buffering off` - Disables request buffering
- `gzip off` - Disables compression for immediate delivery

### Gunicorn Configuration (`Dockerfile`)
- Increased timeout settings for long-running requests
- Optimized worker configuration for streaming

### Application Headers
- `Cache-Control: no-cache` - Prevents client caching
- `Connection: keep-alive` - Maintains persistent connection
- `X-Accel-Buffering: no` - Explicitly disables Nginx buffering

## Troubleshooting SSE Issues

1. **Test without Docker first** - Ensure SSE works locally
2. **Use development setup** - Try `docker-compose.dev.yml` to bypass Nginx
3. **Check browser console** - Look for EventSource connection errors
4. **Monitor container logs** - Watch for buffering or timeout issues

## Environment Variables

Copy `env.sample` to `../.env` and configure:

## Large File Upload Support

The deployment is configured to support large file uploads (up to 500MB by default):

### Configuration Settings

**Nginx Configuration:**
- `client_max_body_size 500M` - Maximum file size allowed
- `client_body_timeout 300s` - Timeout for file upload
- `proxy_request_buffering off` - Stream uploads directly to backend
- `proxy_max_temp_file_size 0` - Disable temp files for uploads

**Gunicorn Configuration:**
- Extended timeout settings for processing large files
- Request line and field size limits increased

### Customizing File Size Limits

To change the maximum file size, edit `deployment/nginx/nginx.conf`:

```nginx
# Change 500M to your desired limit
client_max_body_size 1G;  # For 1GB limit
```

**Important:** After changing the limit:
1. Rebuild the Docker containers: `docker-compose down && docker-compose up -d`
2. Ensure sufficient disk space is available
3. Consider increasing timeouts for very large files

### Supported File Types

The application supports:
- Documents: PDF, TXT, DOCX, CSV
- Archives: ZIP, TAR, RAR (auto-extracted)
- Images: PNG, JPG, GIF, WebP, SVG
- Code files: All text-based formats

### Troubleshooting Large Uploads

If uploads fail:
1. Check Nginx error logs: `docker-compose logs nginx`
2. Verify disk space: `df -h`
3. Monitor upload progress in browser dev tools
4. For files >500MB, increase the `client_max_body_size` setting

## Services

The deployment includes:

- **`app`** - FastAPI application server
- **`nginx`** - Reverse proxy and static file server
- **`postgres`** - PostgreSQL database  
- **`pgadmin`** - Database administration interface

## Ports

- **80/443** - Nginx (HTTP/HTTPS)
- **8000** - Application (direct access)
- **5433** - PostgreSQL database
- **5050** - pgAdmin interface

## Data Persistence

All application data is stored in `../data/` which is mounted as a Docker volume.

## SSL Configuration

Place SSL certificates in `nginx/certs/` and update `nginx/nginx.conf` accordingly.

## Troubleshooting

View logs:
```bash
docker-compose logs -f [service_name]
```

Restart services:
```bash
docker-compose restart [service_name]
``` 