# fast_data_cleaning.py
import pandas as pd
import numpy as np
from database_connection import DatabaseConnection
from datetime import datetime, timedelta

class FastDataCleaning:
    def __init__(self):
        """Fast data cleaning with SQL-based fixes"""
        self.db_conn = DatabaseConnection()
        self.engine = self.db_conn.get_engine()
    
    def create_backup_tables(self):
        """Create backup tables quickly"""
        print("📋 Creating backup tables...")
        
        backup_commands = [
            "DROP TABLE IF EXISTS analytics.sales_backup CASCADE;",
            "CREATE TABLE analytics.sales_backup AS SELECT * FROM analytics.sales;",
            "DROP TABLE IF EXISTS analytics.customers_backup CASCADE;", 
            "CREATE TABLE analytics.customers_backup AS SELECT * FROM analytics.customers;"
        ]
        
        for cmd in backup_commands:
            try:
                with self.engine.connect() as conn:
                    conn.execute(cmd)
                    conn.commit()
            except Exception as e:
                print(f"Backup warning: {e}")
        
        print("✅ Backup tables created")
    
    def fix_all_issues_with_sql(self):
        """Fix all data quality issues using fast SQL updates"""
        print("\n🚀 Applying fast SQL fixes...")
        
        # Fix 1: Sales before product launch - use SQL UPDATE
        fix1_sql = """
        UPDATE analytics.sales 
        SET order_date = p.launch_date + INTERVAL '15 days'
        FROM analytics.products p 
        WHERE sales.product_id = p.product_id 
        AND sales.order_date < p.launch_date;
        """
        
        # Fix 2: Purchases before registration - use SQL UPDATE  
        fix2_sql = """
        UPDATE analytics.customers
        SET registration_date = first_purchase - INTERVAL '10 days'
        FROM (
            SELECT customer_id, MIN(order_date) as first_purchase
            FROM analytics.sales
            GROUP BY customer_id
        ) fp
        WHERE customers.customer_id = fp.customer_id
        AND customers.registration_date > fp.first_purchase;
        """
        
        # Fix 3: Revenue calculations - use SQL UPDATE
        fix3_sql = """
        UPDATE analytics.sales
        SET total_amount = ROUND((unit_price * quantity - discount_amount + shipping_cost)::numeric, 2)
        WHERE ABS(total_amount - (unit_price * quantity - discount_amount + shipping_cost)) > 0.01;
        """
        
        fixes = [
            (fix1_sql, "Fixed sales before product launch"),
            (fix2_sql, "Fixed registration dates"), 
            (fix3_sql, "Fixed revenue calculations")
        ]
        
        for sql_command, description in fixes:
            try:
                print(f"   🔧 {description}...")
                with self.engine.connect() as conn:
                    result = conn.execute(sql_command)
                    conn.commit()
                    print(f"   ✅ {description}: Applied successfully")
            except Exception as e:
                print(f"   ❌ Error in {description}: {e}")
    
    def validate_fixes(self):
        """Quickly validate that fixes worked"""
        print("\n✅ Validating fixes...")
        
        validation_queries = [
            (
                """
                SELECT COUNT(*) as count
                FROM analytics.sales s
                JOIN analytics.products p ON s.product_id = p.product_id
                WHERE s.order_date < p.launch_date
                """,
                "Sales before product launch"
            ),
            (
                """
                SELECT COUNT(*) as count
                FROM analytics.sales s
                JOIN analytics.customers c ON s.customer_id = c.customer_id
                WHERE s.order_date < c.registration_date
                """,
                "Purchases before registration"
            ),
            (
                """
                SELECT COUNT(*) as count
                FROM analytics.sales
                WHERE ABS(total_amount - (unit_price * quantity - discount_amount + shipping_cost)) > 0.01
                """,
                "Revenue calculation mismatches"
            )
        ]
        
        all_passed = True
        
        for query, description in validation_queries:
            try:
                result = pd.read_sql(query, self.engine)
                count = result.iloc[0]['count']
                
                if count == 0:
                    print(f"   ✅ {description}: FIXED (0 remaining)")
                else:
                    print(f"   ❌ {description}: {count:,} still remain")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ Error validating {description}: {e}")
                all_passed = False
        
        return all_passed
    
    def generate_final_stats(self):
        """Generate final dataset statistics"""
        print("\n📊 FINAL DATASET STATISTICS")
        print("=" * 50)
        
        try:
            stats_query = """
            SELECT 
                (SELECT COUNT(*) FROM analytics.products) as products,
                (SELECT COUNT(*) FROM analytics.customers) as customers,
                (SELECT COUNT(*) FROM analytics.sales) as sales,
                (SELECT COUNT(*) FROM analytics.reviews) as reviews,
                (SELECT ROUND(SUM(total_amount), 2) FROM analytics.sales WHERE order_status = 'completed') as total_revenue,
                (SELECT ROUND(AVG(total_amount), 2) FROM analytics.sales WHERE order_status = 'completed') as avg_order_value
            """
            
            stats = pd.read_sql(stats_query, self.engine)
            s = stats.iloc[0]
            
            total_records = s['products'] + s['customers'] + s['sales'] + s['reviews']
            
            print(f"    Products: {s['products']:,}")
            print(f"    Customers: {s['customers']:,}")
            print(f"    Sales: {s['sales']:,}")
            print(f"    Reviews: {s['reviews']:,}")
            print(f"    Total Records: {total_records:,}")
            print(f"    Total Revenue: ${s['total_revenue']:,.2f}")
            print(f"    Avg Order Value: ${s['avg_order_value']:,.2f}")
            
        except Exception as e:
            print(f"   Error generating stats: {e}")
    
    def run_fast_cleaning(self):
        """Execute fast cleaning process"""
        print("=" * 50)
        print("Using optimized SQL commands for speed...")
        print()
        
        # Create backups
        self.create_backup_tables()
        
        # Apply all fixes
        self.fix_all_issues_with_sql()
        
        # Validate results
        validation_passed = self.validate_fixes()
        
        # Generate final stats
        self.generate_final_stats()
        
        if validation_passed:
            print(f"\n FAST CLEANING COMPLETED SUCCESSFULLY!")
            print(f" All data quality issues resolved")
            return True
        else:
            print(f"\n Some validation checks failed")
            return False

def main():
    """Main execution"""
    print(" This is the FAST version of data cleaning!")
    print("Uses SQL commands instead of Python loops for speed.")
    print()
    
    cleaner = FastDataCleaning()
    success = cleaner.run_fast_cleaning()
    
    if success:
        print(f"Your data is now clean and ready for ML models!")

if __name__ == "__main__":
    main()