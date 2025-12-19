```
import os
from dotenv import load_dotenv

# Load variables immediately when this module is imported
load_dotenv()

def get_db_connection_str():
    """
    Constructs and returns the database connection string 
    using variables from the .env file.
    """
    return (
        f"{os.getenv('DB_TYPE')}://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
        f"/{os.getenv('DB_NAME')}"
    )
```

```
import os
import connectorx as cx
import pyarrow.parquet as pq

def export_to_parquet(connection_str, query, file_name):
    """
    Executes a SQL query and saves the result as a Parquet file 
    inside the 'EE_file' folder.
    
    Parameters:
    - connection_str: The full DB connection URL.
    - query: SQL query string.
    - file_name: Name of the output file (e.g., 'data.parquet').
    """
    
    # 1. Define output directory and ensure it exists
    output_dir = "EE_file"
    
    # This creates the folder if it doesn't exist; does nothing if it does.
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Create the full file path
    output_path = os.path.join(output_dir, file_name)
    
    print(f"Executing query and saving to: {output_path}...")

    try:
        # 3. Fetch data directly as an Arrow Table (Fastest method)
        # We explicitly enforce return_type="arrow" as requested
        arrow_table = cx.read_sql(connection_str, query, return_type="arrow")
        
        # 4. Write the Arrow table to Parquet
        pq.write_table(arrow_table, output_path, compression="snappy")
        
        print("Success! File saved.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
```


```
# Import from your custom modules
from config.settings import get_db_connection_str
from src.utils import export_to_parquet

def main():
    # 1. Get the connection string from config
    db_conn = get_db_connection_str()
    
    # 2. Define your query
    my_query = "SELECT * FROM my_table LIMIT 1000"
    
    # 3. Define the output filename
    parquet_name = "daily_extract.parquet"
    
    # 4. Call the global function
    # This will fetch data and save it into EE_file/daily_extract.parquet
    export_to_parquet(
        connection_str=db_conn,
        query=my_query,
        file_name=parquet_name
    )

if __name__ == "__main__":
    main()
```
