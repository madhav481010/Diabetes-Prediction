pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building the project...'
                sh 'mvn clean package'
            }
        }

        stage('Test') {
            steps {
                echo 'Running tests...'
                sh 'mvn test'
            }
        }

        stage('Deploy to Staging') {
            steps {
                echo 'Deploying to staging...'
                sh './deploy_staging.sh'
            }
        }

        stage('Deploy to Production') {
            steps {
                echo 'Deploying to production...'
                sh './deploy_prod.sh'
            }
        }
    }
}
