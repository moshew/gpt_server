# Docker Instructions

This document explains how to run the application using Docker.

## Prerequisites

- Docker and Docker Compose installed on your system
- Azure OpenAI API key and endpoint URL

## Setup

1. Copy the sample environment file and fill in your own values:

```bash
cp .env.sample .env
```

2. Edit the `.env` file with appropriate values for your environment:
   - Replace `your_azure_api_key` with your actual Azure API key
   - Replace `your_azure_endpoint` with your Azure endpoint URL
   - Configure other settings as needed
   - The database URL is already configured for the Docker setup

## Running the Application

Start the application using Docker Compose:

```bash
docker-compose up -d
```

This will:
- Build the Docker image if needed
- Start a PostgreSQL database container
- Start the API server
- Connect the API to the database

The API will be available at http://localhost:8000.

## Stopping the Application

To stop the application:

```bash
docker-compose down
```

To stop the application and remove all data (including the database volume):

```bash
docker-compose down -v
```

## Viewing Logs

To view logs from the application:

```bash
docker-compose logs -f api
```

## Rebuilding the Image

If you make changes to the code and need to rebuild the image:

```bash
docker-compose up -d --build
``` 