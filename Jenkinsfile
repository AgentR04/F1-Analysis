pipeline {
    agent any

    environment {
        PROJECT_ID    = "crypto-lore-468803-u3"
        REGION        = "us-central1"
        REPO_NAME     = "f1-repo"
        IMAGE_NAME    = "f1-analysis-app"

        // Jenkins credential IDs
        GCP_CREDS      = credentials('gcp-sa-key')   // Service account key file
        SLACK_WEBHOOK  = credentials('slack-webhook') // Secret text
    }

    stages {

        /* ------------------ Pull Source Code ------------------ */
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        /* ------------------ Generate Image Tag ------------------ */
        stage('Set Image Tag') {
            steps {
                script {
                    IMAGE_TAG = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    echo "Using Image Tag: ${IMAGE_TAG}"
                }
            }
        }

        /* ------------------ Authenticate GCP ------------------ */
        stage('Authenticate to GCP') {
            steps {
                withCredentials([file(credentialsId: 'gcp-sa-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    sh '''
                        gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
                        gcloud config set project $PROJECT_ID
                        gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
                    '''
                }
            }
        }

        /* ------------------ Build Docker Image ------------------ */
        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
            }
        }

        /* ------------------ Security Scan (Fail if Critical) ------------------ */
        stage('Security Scan - Trivy') {
            steps {
                sh """
                    trivy image --exit-code 1 --severity HIGH,CRITICAL ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        /* ------------------ Push Image to Artifact Registry ------------------ */
        stage('Push Image') {
            steps {
                sh """
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
                    docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        /* ------------------ Deploy to GKE ------------------ */
        stage('Deploy to GKE') {
            steps {
                sh """
                    gcloud container clusters get-credentials f1-cluster --region=${REGION} --project=${PROJECT_ID}

                    kubectl set image deployment/f1-app \
                    f1-app=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

                    kubectl rollout status deployment/f1-app
                """
            }
        }
    }

    /* ------------------ Notifications ------------------ */
    post {
        success {
            echo "Deployment Successful!"
            sh """
                curl -X POST -H 'Content-type: application/json' \
                --data "{\\"text\\": \\"✔ SUCCESS: F1 App Deployed | Tag: ${IMAGE_TAG}\\"}" \
                $SLACK_WEBHOOK
            """
        }

        failure {
            echo "Pipeline Failed!"
            sh """
                curl -X POST -H 'Content-type: application/json' \
                --data "{\\"text\\": \\"❌ FAILURE: Build/Deploy Failed | Check Jenkins Logs\\"
                }" \
                $SLACK_WEBHOOK
            """
        }
    }
}
