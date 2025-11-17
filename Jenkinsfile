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

        stage('Start Minikube') {
            steps {
                script {
                    echo "[INFO] Checking Minikube status..."

                    def status = sh(
                        script: "minikube status --format='{{.Host}}'",
                        returnStdout: true
                    ).trim()

                    if (!status.contains("Running")) {
                        echo "[INFO] Minikube not running — starting..."
                        sh """
                            sudo minikube start --driver=docker --memory=3000mb
                        """
                    } else {
                        echo "[INFO] Minikube already running."
                    }
                }
            }
        }

        stage('Configure Docker to use Minikube') {
            steps {
                sh """
                    echo "[INFO] Switching Docker to Minikube environment..."
                    eval \$(minikube -p minikube docker-env)
                """
            }
        }

        stage('Build Docker Image') {
            steps {
                sh """
                    echo "[INFO] Building Docker image inside Minikube..."
                    eval \$(minikube -p minikube docker-env)
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                """
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh """
                    echo "[INFO] Updating deployment image..."
                    sed -i "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}:${IMAGE_TAG}|g" k8s/deployment.yaml

                    echo "[INFO] Applying Kubernetes manifests..."
                    kubectl apply -f k8s/deployment.yaml
                    kubectl apply -f k8s/service.yaml

                    echo "[INFO] Waiting for rollout..."
                    kubectl rollout status deployment/f1-app --timeout=90s
                """
            }
        }

        stage('Show App URL') {
            steps {
                script {
                    def url = sh(script: "minikube service f1-app-service --url", returnStdout: true).trim()
                    echo "🎉 APP DEPLOYED SUCCESSFULLY!"
                    echo "👉 Access your app at: ${url}"
                }
            }
        }
    }

    post {
        failure {
            echo "❌ Pipeline Failed — Check Logs"
        }
        success {
            echo "✅ Pipeline Successful!"
        }
    }
}

