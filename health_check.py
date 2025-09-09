#!/usr/bin/env python3
"""
Health check endpoint for Railway deployment
This simple Flask app provides a health check endpoint
that Railway can use to monitor the application status.
"""

from flask import Flask, jsonify
import os
import sys

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint for Railway monitoring"""
    try:
        # Check if required environment variables are set
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_key:
            return jsonify({
                'status': 'unhealthy',
                'error': 'OPENAI_API_KEY not set'
            }), 500
        
        # Check if we can import required modules
        try:
            import streamlit
            import langchain
            import chromadb
        except ImportError as e:
            return jsonify({
                'status': 'unhealthy',
                'error': f'Missing dependency: {str(e)}'
            }), 500
        
        return jsonify({
            'status': 'healthy',
            'message': 'RAG Assistant is running',
            'python_version': sys.version,
            'streamlit_version': streamlit.__version__
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/')
def root():
    """Root endpoint redirects to Streamlit app"""
    return jsonify({
        'message': 'RAG Assistant Health Check',
        'status': 'running',
        'streamlit_app': '/streamlit'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)