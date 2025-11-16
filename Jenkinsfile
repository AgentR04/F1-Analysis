pipeline {
    agent any

    environment {
        IMAGE_NAME = "f1-analysis-app"
        IMAGE_TAG  = ""
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
                    IMAGE_TAG = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    echo "Using Image Tag: ${IMAGE_TAG}"
                }
            }
        }

        stage('Configure Docker for Minikube') {
            steps {
                sh '''
                    eval $(minikube docker-env)
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    eval $(minikube docker-env)
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                '''
            }
        }

        stage('Security Scan - Trivy') {
            steps {
                sh '''
                    trivy image --exit-code 1 --severity HIGH,CRITICAL ${IMAGE_NAME}:${IMAGE_TAG} || true
                '''
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh '''
                    sed -i "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}:${IMAGE_TAG}|g" k8s/deployment.yaml
                    kubectl apply -f k8s/deployment.yaml
                    kubectl apply -f k8s/service.yaml
                    kubectl rollout status deployment/f1-app
                '''
            }
        }
    }

    post {
        success {
            echo "🚀 Deployment to Minikube Successful!"
        }
        failure {
            echo "❌ Pipeline Failed — check logs!"
        }
    }
}

