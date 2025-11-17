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

        stage('Start Minikube if not running') {
            steps {
                script {
                    echo "[INFO] Checking Minikube status..."
                    def status = sh(script: "minikube status --format='{{.Host}}'", returnStdout: true).trim()

                    if (status != "Running") {
                        echo "[INFO] Starting Minikube..."
                        sh "minikube start --driver=docker"
                    } else {
                        echo "[INFO] Minikube already running"
                    }
                }
            }
        }

        stage('Configure Docker to use Minikube') {
            steps {
                script {
                    echo "[INFO] Configuring Docker with Minikube env"
                    sh "eval \$(minikube docker-env)"
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
                sh "trivy image --exit-code 0 ${IMAGE_NAME}:${IMAGE_TAG} || true"
            }
        }

        stage('Deploy to Minikube') {
            steps {
                script {
                    sh """
                        sed -i 's|IMAGE_PLACEHOLDER|${IMAGE_NAME}:${IMAGE_TAG}|g' k8s/deployment.yaml
                        kubectl apply -f k8s/deployment.yaml
                        kubectl apply -f k8s/service.yaml
                        kubectl rollout status deployment/f1-app
                    """
                }
            }
        }

        stage('Show App URL') {
            steps {
                script {
                    def url = sh(script: "minikube service f1-app --url", returnStdout: true).trim()
                    echo "\\n🚀 Your App is Live: ${url}"
                }
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment Completed Successfully!"
        }
        failure {
            echo "❌ Pipeline Failed — Check Logs"
        }
    }
}

