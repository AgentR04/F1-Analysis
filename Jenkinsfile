pipeline {
    agent any

    environment {
        IMAGE_NAME = "f1-analysis-app"
    }

    stages {

        // 1️⃣ Get project source code
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        // 2️⃣ Create unique image tag using git commit
        stage('Set Image Tag') {
            steps {
                script {
                    env.IMAGE_TAG = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    echo "Using Image Tag: ${env.IMAGE_TAG}"
                }
            }
        }

        // 3️⃣ Point Docker to Minikube's Docker daemon
        stage('Connect Docker to Minikube') {
            steps {
                sh '''
                    echo "[INFO] Switching Docker to Minikube environment..."
                    eval $(minikube -p minikube docker-env)
                    docker info
                '''
            }
        }

        // 4️⃣ Build Docker image inside Minikube
        stage('Build Docker Image') {
            steps {
                sh '''
                    echo "[INFO] Building Docker image..."
                    eval $(minikube -p minikube docker-env)
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                '''
            }
        }

        // 5️⃣ Optional image scan (kept but not blocking)
        stage('Security Scan - Trivy') {
            steps {
                sh '''
                    echo "[INFO] Running security scan (ignoring failure)"
                    trivy image --severity HIGH,CRITICAL ${IMAGE_NAME}:${IMAGE_TAG} || true
                '''
            }
        }

        // 6️⃣ Deploy to Minikube using Kubernetes manifests
        stage('Deploy to Minikube') {
            steps {
                sh '''
                    echo "[INFO] Updating Kubernetes deployment manifest..."
                    sed -i "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}:${IMAGE_TAG}|g" k8s/deployment.yaml

                    echo "[INFO] Applying manifests..."
                    kubectl apply -f k8s/deployment.yaml
                    kubectl apply -f k8s/service.yaml

                    echo "[INFO] Checking rollout status..."
                    kubectl rollout status deployment/f1-app
                '''
            }
        }

        // 7️⃣ Show access URL
        stage('Show Access Info') {
            steps {
                sh '''
                    echo "[INFO] Fetching Minikube Service URL..."
                    minikube service f1-app --url || true
                '''
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment Successful! App should be accessible now."
        }
        failure {
            echo "❌ Pipeline Failed — debug required."
        }
    }
}

