# Deployment Guide

This directory contains all the necessary files for deploying the GPT Server application.

## Files Overview

- **`docker-compose.yml`** - Main deployment configuration
- **`Dockerfile`** - Application container definition  
- **`.dockerignore`** - Files to exclude from Docker context
- **`requirements.txt`** - Python dependencies
- **`env.sample`** - Environment variables template
- **`docker-instructions.md`** - Detailed deployment instructions
- **`nginx/`** - Nginx reverse proxy configuration
- **`init-db/`** - Database initialization scripts

## Quick Deployment

1. **Copy environment file**:
   ```bash
   cp env.sample ../.env
   ```

2. **Edit configuration**:
   ```bash
   nano ../.env  # Configure your API keys and database settings
   ```

3. **Deploy with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Check status**:
   ```bash
   docker-compose ps
   ```

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