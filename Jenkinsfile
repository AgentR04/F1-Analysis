pipeline {
    agent any

    environment {
        PROJECT_ID = "devops-mini-project-477719"
        REGION     = "us-central1"
        REPO_NAME  = "f1-repo"
        IMAGE_NAME = "f1-analysis-app"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set Image Tag') {
            steps {
                script {
                    env.IMAGE_TAG = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    echo "Using Image Tag: ${env.IMAGE_TAG}"
                }
            }
        }

        stage('Authenticate to GCP') {
            steps {
                withCredentials([file(credentialsId: 'gcp-sa-key', variable: 'GCLOUD_KEY')]) {
                    sh """
                        gcloud auth activate-service-account --key-file=$GCLOUD_KEY
                        gcloud config set project $PROJECT_ID
                        gcloud auth configure-docker ${REGION}-docker.pkg.dev -q
                    """
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
            }
        }

        stage('Security Scan - Trivy') {
            steps {
                sh "trivy image --exit-code 1 --severity HIGH,CRITICAL ${IMAGE_NAME}:${IMAGE_TAG} || true"
            }
        }

        stage('Push Image') {
            steps {
                sh """
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
                    docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        stage('Deploy to GKE') {
            steps {
                sh """
                    gcloud container clusters get-credentials f1-cluster --region=$REGION --project=$PROJECT_ID
                    sed -i "s|IMAGE_PLACEHOLDER|${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}|g" k8s/deployment.yaml
                    kubectl apply -f k8s/deployment.yaml
                    kubectl apply -f k8s/service.yaml
                    kubectl rollout status deployment/f1-app
                """
            }
        }
    }

    post {
        success {
            echo "Deployment Successful!"
        }
        failure {
            echo "❌ Pipeline Failed — check logs"
        }
    }
}
