import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

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

# Table the scraper writes and the training script reads
TABLE_NAME = 'matches_data'

# Local copy of the scraped data, used for training when there is no database (e.g. CI)
DATA_DIR = Path(os.getenv('DATA_DIR', Path(__file__).parent / 'data'))
COMBINED_CSV = DATA_DIR / 'matches_data.csv'

# API Keys
API_KEYS = {
    'API_KEY': os.getenv("API_KEY"),

}

def get_db_connection():
    """Helper function to get database connection"""
    import mysql.connector
    return mysql.connector.connect(
        user=DB_CONFIG['username'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        port=int(DB_CONFIG['port']),
        database=DB_CONFIG['database'],
    )


def get_sqlalchemy_engine():
    """Helper function to get SQLAlchemy engine for pandas operations"""
    username = quote_plus(DB_CONFIG['username'])
    password = quote_plus(DB_CONFIG['password'] or '')
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    database = DB_CONFIG['database']

    connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    return create_engine(connection_string)
