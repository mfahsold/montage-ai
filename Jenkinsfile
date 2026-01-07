pipeline {
  agent any

  options {
    timestamps()
    ansiColor('xterm')
  }

  environment {
    CI = 'true'
    PY_VER_MINOR = '3.10'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup') {
      steps {
        sh 'chmod +x scripts/ci.sh'
      }
    }

    stage('CI') {
      steps {
        sh './scripts/ci.sh'
      }
    }
  }

  post {
    always {
      junit allowEmptyResults: true, testResults: 'tests/**/junit-*.xml'
      archiveArtifacts allowEmptyArchive: true, artifacts: 'benchmark_results/**/*, data/output/**/*'
    }
  }
}
