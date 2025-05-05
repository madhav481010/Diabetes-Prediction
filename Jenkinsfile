pipeline {
    agent any

    stages {
        stage('Install Python Dependencies') {
            steps {
                bat 'python -m pip install --upgrade pip'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Job 1: Preprocess Data') {
            steps {
                bat 'python preprocess.py'
            }
        }

        stage('Job 2: Train and Evaluate Models') {
            steps {
                bat 'python train_and_evaluate.py'
            }
        }

        stage('Job 3: Deploy and Predict') {
            steps {
                bat 'python deploy.py'
            }
        }
    }

    post {
        success {
            echo '✅ All stages completed successfully.'
        }
        failure {
            echo '❌ Pipeline failed.'
        }
    }
}
