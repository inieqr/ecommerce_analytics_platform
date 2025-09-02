# database_connection.py
import psycopg2
from sqlalchemy import create_engine
import pandas as pd

class DatabaseConnection:
    def __init__(self):
        # pgAdmin CREDENTIALS
        self.config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'ecommerce_analytics',
            'user': 'postgres',       
            'password': '1998Strong' 
        }
    
    def get_connection_string(self):
        return (f"postgresql://{self.config['user']}:{self.config['password']}@"
                f"{self.config['host']}:{self.config['port']}/{self.config['database']}")
    
    def get_engine(self):
        return create_engine(self.get_connection_string())
    
    def test_connection(self):
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                result = pd.read_sql("SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'analytics'", conn)
                print(f"Database connection successful! Found {result.iloc[0]['table_count']} analytics tables.")
                return True
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Please check your username and password in the config above.")
            return False

# Connection test
if __name__ == "__main__":
    db = DatabaseConnection()
    db.test_connection()