#!/bin/bash

# Function to deploy a service
deploy_service() {
  local service_name=$1
  local dockerfile_path=$2
  local image_name=$3
  local environment=$4
  local tag=$5

  echo "----------------------------------------------"
  echo "Deploying $service_name to $environment environment"
  echo "----------------------------------------------"

  # Build the Docker image
  docker build -f $dockerfile_path -t $image_name .

  # Tag the image
  docker tag $image_name:latest gcr.io/peoples-software/$service_name:$tag

  # Push the image to Google Container Registry
  docker push gcr.io/peoples-software/$service_name:$tag

  echo "$service_name deployed successfully to $environment"
  echo "----------------------------------------------"
}

# Deploy to production
deploy_to_prod() {
  echo "Starting production deployment..."

  # Clustering service
  deploy_service "clustering-service" "clustering/Dockerfile" "peeps-clustering-job" "prod" "prod"

  # Algo pipeline executer service
  deploy_service "algo-pipeline-executer-service" "algo_pipeline_executer/Dockerfile" "peeps-algo_pipeline_executer-job" "prod" "prod"

  # Preprocessing service
  deploy_service "preprocessing-service" "preprocessing/Dockerfile" "peeps-preprocess-job" "prod" "prod"

  # Indexing service
  deploy_service "indexing-service" "vector_indexing/Dockerfile" "peeps-indexing-job" "prod" "prod"

  echo "Production deployment completed!"
}

# Deploy to development
deploy_to_dev() {
  echo "Starting development deployment..."

  # Clustering service
  deploy_service "clustering-service" "clustering/Dockerfile" "peeps-clustering-job-dev" "dev" "dev"

  # Preprocessing service
  deploy_service "preprocessing-service" "preprocessing/Dockerfile" "peeps-preprocess-job-dev" "dev" "dev"

  # Indexing service
  deploy_service "indexing-service" "vector_indexing/Dockerfile" "peeps-indexing-job-dev" "dev" "dev"

  echo "Development deployment completed!"
}

# Main script logic to choose environment
if [ "$1" == "prod" ]; then
  deploy_to_prod
elif [ "$1" == "dev" ]; then
  deploy_to_dev
else
  echo "Usage: ./deploy.sh [prod|dev]"
  exit 1
fi
