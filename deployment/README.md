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

```bash
# View application logs
docker-compose logs -f app

# View nginx logs
docker-compose logs -f nginx
```

## Environment Variables

Copy `env.sample` to `../.env` and configure:

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