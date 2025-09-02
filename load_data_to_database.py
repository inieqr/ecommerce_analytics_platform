# load_data_to_database.py
import pandas as pd
from database_connection import DatabaseConnection
import os
from tqdm import tqdm
import numpy as np

def load_data_to_database():
    print(" Loading E-commerce Data into PostgreSQL Database")
    print("=" * 60)
    
    # Connect to database
    print("1. Connecting to database...")
    db_conn = DatabaseConnection()
    engine = db_conn.get_engine()
    
    # Test connection first
    if not db_conn.test_connection():
        print(" Database connection failed. Please check your credentials.")
        return False
    
    print("✅ Database connection successful!")
    
    # Define files to load in order (important for foreign keys)
    files_to_load = [
        ('data/products.csv', 'products'),
        ('data/customers.csv', 'customers'),
        ('data/sales.csv', 'sales'), 
        ('data/reviews.csv', 'reviews')
    ]
    
    print(f"\n2. Loading {len(files_to_load)} datasets...")
    
    for file_path, table_name in files_to_load:
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            print(f"   Make sure you ran the data generation first!")
            continue
            
        print(f"\n Loading {file_path} → analytics.{table_name}")
        
        try:
            # Read CSV file
            print(f"   Reading CSV file...")
            df = pd.read_csv(file_path)
            print(f"    Read {len(df):,} records from CSV")
            
            # Data cleaning and type conversion
            print(f"    Processing data types...")
            
            # Convert date columns to proper format
            date_columns = ['launch_date', 'registration_date', 'order_date', 'review_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                    print(f"      Converted {col} to date format")
            
            # Handle any NaN values in numeric columns
            numeric_columns = ['price', 'total_amount', 'unit_price', 'discount_amount', 'shipping_cost', 'rating']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with 0 for numeric columns
                    df[col] = df[col].fillna(0)
            
            # Clean text fields (remove any problematic characters)
            text_columns = ['title', 'review_text', 'shipping_address']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
            
            print(f"    Data processing complete")
            
            # Load to database in batches for better performance
            print(f"    Loading to database in batches...")
            batch_size = 5000
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            successful_records = 0
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(df))
                batch = df.iloc[start_idx:end_idx].copy()
                
                try:
                    # Load batch to database
                    batch.to_sql(
                        table_name, 
                        engine, 
                        schema='analytics',
                        if_exists='append', 
                        index=False, 
                        method='multi'
                    )
                    successful_records += len(batch)
                    
                    # Progress update
                    if batch_num % 5 == 0 or batch_num == total_batches - 1:
                        progress = (batch_num + 1) / total_batches * 100
                        print(f"      Progress: {progress:.1f}% ({successful_records:,}/{len(df):,} records)")
                        
                except Exception as batch_error:
                    print(f"        Error in batch {batch_num + 1}: {batch_error}")
                    continue
            
            print(f"    Successfully loaded {successful_records:,} records into analytics.{table_name}")
            
        except Exception as e:
            print(f"    Error loading {table_name}: {e}")
            continue
    
    print(f"\n3. Verifying data loading...")
    
    # Verify data was loaded correctly
    verification_queries = [
        ("Products", "SELECT COUNT(*) as count FROM analytics.products"),
        ("Customers", "SELECT COUNT(*) as count FROM analytics.customers"), 
        ("Sales", "SELECT COUNT(*) as count FROM analytics.sales"),
        ("Reviews", "SELECT COUNT(*) as count FROM analytics.reviews")
    ]
    
    total_loaded = 0
    
    for table_name, query in verification_queries:
        try:
            result = pd.read_sql(query, engine)
            count = result.iloc[0]['count']
            total_loaded += count
            print(f"   {table_name}: {count:,} records")
        except Exception as e:
            print(f"    Error checking {table_name}: {e}")
    
    print(f"\n" + "=" * 60)
    print(" DATA LOADING COMPLETE!")
    print("=" * 60)
    print(f" Total Records Loaded: {total_loaded:,}")
    print(f" Database: ecommerce_analytics")
    print(f" Schema: analytics")
    print(f" Tables: products, customers, sales, reviews")
    print()
    print(" Your database is now ready for analytics!")
   
    
    return True

def verify_data_relationships():
    """Verify that the data relationships are working correctly"""
    print("\n Verifying Data Relationships...")
    
    db_conn = DatabaseConnection()
    engine = db_conn.get_engine()
    
    # Test queries to verify relationships
    test_queries = [
        ("Sales with Products", """
            SELECT COUNT(*) as count
            FROM analytics.sales s
            JOIN analytics.products p ON s.product_id = p.product_id
            LIMIT 5
        """),
        ("Reviews with Customers", """
            SELECT COUNT(*) as count  
            FROM analytics.reviews r
            JOIN analytics.customers c ON r.customer_id = c.customer_id
            LIMIT 5
        """),
        ("Sample Business Data", """
            SELECT 
                p.category,
                COUNT(s.sale_id) as total_sales,
                SUM(s.total_amount) as revenue,
                AVG(r.rating) as avg_rating
            FROM analytics.products p
            LEFT JOIN analytics.sales s ON p.product_id = s.product_id
            LEFT JOIN analytics.reviews r ON p.product_id = r.product_id
            WHERE s.order_status = 'completed'
            GROUP BY p.category
            ORDER BY revenue DESC
            LIMIT 5
        """)
    ]
    
    for test_name, query in test_queries:
        try:
            result = pd.read_sql(query, engine)
            print(f"    {test_name}: Working correctly")
            if test_name == "Sample Business Data":
                print("      Top categories by revenue:")
                for _, row in result.iterrows():
                    print(f"        {row['category']}: ${row['revenue']:,.2f} revenue, {row['avg_rating']:.1f}★ rating")
        except Exception as e:
            print(f"    {test_name}: {e}")

def main():
    """Main execution function"""
    success = load_data_to_database()
    
    if success:
        verify_data_relationships()
        
        print(f"\nVerified:")
    else:
        print(f"\n Data loading failed.")

if __name__ == "__main__":
    main()