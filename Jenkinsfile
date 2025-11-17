pipeline {
    agent any

    environment {
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

        stage('Configure Docker to use Minikube') {
            steps {
                sh '''
                    echo "[INFO] Using Minikube Docker..."
                    eval $(minikube docker-env)
                    docker system info
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    echo "[INFO] Building Docker image..."
                    eval $(minikube docker-env)
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                '''
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh '''
                    echo "[INFO] Updating deployment manifest..."
                    sed -i "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}:${IMAGE_TAG}|g" k8s/deployment.yaml

                    echo "[INFO] Applying to Kubernetes..."
                    kubectl apply -f k8s/deployment.yaml
                    kubectl apply -f k8s/service.yaml

                    echo "[INFO] Waiting for rollout..."
                    kubectl rollout status deployment/f1-app
                '''
            }
        }

        stage('Show App URL') {
            steps {
                sh '''
                    minikube service f1-app-service --url
                '''
            }
        }
    }

    post {
        success {
            echo "🎉 SUCCESS — App deployed on Minikube!"
        }
        failure {
            echo "❌ Pipeline Failed — check logs"
        }
    }
}

