pipeline {
    agent any

    environment {
        PROJECT_ID = 'crypto-lore-468803-u3'
        IMAGE_NAME = 'f1-analysis-app'
        IMAGE_TAG = 'latest'
        REGION = 'us-central1-a' 
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/AgentR04/F1-Analysis.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
            }
        }

        stage('Security Scan') {
            steps {
                sh 'trivy image $IMAGE_NAME:$IMAGE_TAG || true'
            }
        }

        stage('Push to Artifact Registry') {
            steps {
                sh '''
                    gcloud auth configure-docker $REGION-docker.pkg.dev -q
                    docker tag $IMAGE_NAME:$IMAGE_TAG $REGION-docker.pkg.dev/$PROJECT_ID/f1-repo/$IMAGE_NAME:$IMAGE_TAG
                    docker push $REGION-docker.pkg.dev/$PROJECT_ID/f1-repo/$IMAGE_NAME:$IMAGE_TAG
                '''
            }
        }

        stage('Deploy to GKE') {
            steps {
                sh '''
                    gcloud container clusters get-credentials f1-cluster --region=$REGION --project=$PROJECT_ID
                    kubectl apply -f k8s-deployment.yaml
                '''
            }
        }
    }
}
