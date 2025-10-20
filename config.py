# config.py - DO NOT commit this file to version control
# Add this file to .gitignore

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'username': os.getenv('DB_USERNAME', 'root'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'port': os.getenv('DB_PORT', '3306'),
    'database': os.getenv('DB_NAME', 'footballML')
}

# API Keys
API_KEYS = {
    'API_KEY': os.getenv("API_KEY"),
    
}

def get_db_connection():
    """Helper function to get database connection"""
    import mysql.connector
    return mysql.connector.connect(**DB_CONFIG)


def get_sqlalchemy_engine():
    """Helper function to get SQLAlchemy engine for pandas operations"""
    username = DB_CONFIG['username']
    password = DB_CONFIG['password']
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    database = DB_CONFIG['database']
    
    connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    return create_engine(connection_string)