
# Automated Deployment Script for Peeps Services

We have introduced an automated deployment process to streamline the deployment of services in both **production** and **development** environments. The deployment is managed through a Bash script that builds, tags, and pushes Docker images for various services to the Google Container Registry.

## Usage

To use the deployment script, follow these steps:

1. **Make the script executable** (only needs to be done once):
   ```bash
   chmod +x deploy.sh
   ```

2. **Deploy to production**:
   Run the script with the `prod` argument to deploy all services to the production environment.
   ```bash
   ./deploy.sh prod
   ```

3. **Deploy to development**:
   Run the script with the `dev` argument to deploy all services to the development environment.
   ```bash
   ./deploy.sh dev
   ```

## Services Deployed

The script automates the deployment of the following services:
- **Clustering Service**
- **Algo Pipeline Executer Service**
- **Preprocessing Service**
- **Indexing Service**

Each service is built using its respective Dockerfile and pushed to Google Container Registry with the appropriate tags for either production or development.

## Example

For example, to deploy the preprocessing service to the development environment:
```bash
./deploy.sh dev
```
This will:
- Build the Docker image for the preprocessing service.
- Tag the image for development.
- Push the image to the Google Container Registry under the development tag.

## Benefits

- **Time-Saving**: The script automates repetitive steps and reduces the risk of manual errors.
- **Consistency**: Ensures that all services are deployed in a consistent manner for both environments.
- **Scalable**: Easily extendable if new services are added to the project in the future.
