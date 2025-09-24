#!/bin/bash

# Quick build and push script for NL2SQL Agent API
set -e

# Load environment variables
if [ -f "src/.env" ]; then
    while IFS= read -r line; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]]; then
            continue
        fi
        # Remove inline comments and export
        clean_line=$(echo "$line" | sed 's/#.*$//' | xargs)
        if [[ "$clean_line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$clean_line"
        fi
    done < src/.env
fi

# Set defaults
CONTAINER_IMAGE_NAME=${CONTAINER_IMAGE_NAME:-"nl2sql-agent-api"}
CONTAINER_IMAGE_TAG=${1:-${CONTAINER_IMAGE_TAG:-"latest"}}

# Check required variables
if [[ -z "$CONTAINER_REGISTRY_NAME" ]]; then
    echo "ERROR: CONTAINER_REGISTRY_NAME must be set"
    exit 1
fi

# Get registry login server
REGISTRY_LOGIN_SERVER=$(az acr show --name $CONTAINER_REGISTRY_NAME --query loginServer --output tsv)

echo "Building and pushing image: $REGISTRY_LOGIN_SERVER/$CONTAINER_IMAGE_NAME:$CONTAINER_IMAGE_TAG"

# Login to ACR
az acr login --name $CONTAINER_REGISTRY_NAME

# Build and push (targeting linux/amd64 for Azure compatibility)
docker build --platform linux/amd64 -t $REGISTRY_LOGIN_SERVER/$CONTAINER_IMAGE_NAME:$CONTAINER_IMAGE_TAG .
docker push $REGISTRY_LOGIN_SERVER/$CONTAINER_IMAGE_NAME:$CONTAINER_IMAGE_TAG

echo "✅ Image pushed successfully!"
echo "Image: $REGISTRY_LOGIN_SERVER/$CONTAINER_IMAGE_NAME:$CONTAINER_IMAGE_TAG"
