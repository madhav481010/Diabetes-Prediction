pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building the project...'
                bat 'mvn clean package'
            }
        }

        stage('Test') {
            steps {
                echo 'Running tests...'
                bat 'mvn test'
            }
        }

        stage('Deploy to Staging') {
            steps {
                echo 'Deploying to staging...'
                bat 'deploy_staging.bat'
            }
        }

        stage('Deploy to Production') {
            steps {
                echo 'Deploying to production...'
                bat 'deploy_prod.bat'
            }
        }
    }
}
