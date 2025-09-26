from db_config import get_db_connection

def test_connection():
    try:
        conn = get_db_connection()
        print("Database connection successful!")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {str(e)}")

if __name__ == "__main__":
    test_connection()