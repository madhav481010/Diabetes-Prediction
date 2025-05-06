pipeline {
    agent any

    stages {
        stage('Install Python Dependencies') {
            steps {
                bat '"C:\\Users\\hp1\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pip install --upgrade pip'
                bat '"C:\\Users\\hp1\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pip install -r requirements.txt'
            }
        }

        stage('Job 1: Preprocess Data') {
            steps {
                bat '"C:\\Users\\hp1\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" preprocess.py'
            }
        }

        stage('Job 2: Train and Evaluate Models') {
            steps {
                bat '"C:\\Users\\hp1\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" train_and_evaluate.py'
            }
        }

        stage('Job 3: Deploy and Predict') {
            steps {
                bat '"C:\\Users\\hp1\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" deploy.py'
            }
        }

        stage('Start Flask Server') {
            steps {
                bat '"C:\\Users\\hp1\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" app.py'
            }
        }

    }

    post {
        success {
            echo '✅ Pipeline completed successfully.'
        }
        failure {
            echo '❌ Pipeline failed.'
        }
    }
}
