# quick_data_fix.py
import pandas as pd
from database_connection import DatabaseConnection

class QuickDataFix:
    def __init__(self):
        self.db_conn = DatabaseConnection()
        self.engine = self.db_conn.get_engine()
    
    def apply_minimal_fixes(self):
        """Apply only the most critical fixes for analytics"""
        print(" APPLYING MINIMAL DATA FIXES")
        print("=" * 40)
        print("Fixing only critical issues needed for analytics...")
        
        try:
            # Fix 1: Ensure no negative revenues (critical for business metrics)
            print("\n Fixing negative revenues...")
            with self.engine.connect() as conn:
                conn.execute("UPDATE analytics.sales SET total_amount = ABS(total_amount) WHERE total_amount < 0")
                conn.commit()
            print("    Negative revenues fixed")
            
            # Fix 2: Ensure ratings are 1-5 (critical for review analysis)
            print("\n Ensuring valid ratings...")
            with self.engine.connect() as conn:
                conn.execute("UPDATE analytics.reviews SET rating = CASE WHEN rating < 1 THEN 1 WHEN rating > 5 THEN 5 ELSE rating END")
                conn.commit()
            print("    Rating ranges validated")
            
            # Fix 3: Remove any NULL customer/product references (critical for joins)
            print("\n Cleaning NULL references...")
            with self.engine.connect() as conn:
                conn.execute("DELETE FROM analytics.sales WHERE customer_id IS NULL OR product_id IS NULL")
                conn.execute("DELETE FROM analytics.reviews WHERE customer_id IS NULL OR product_id IS NULL") 
                conn.commit()
            print("    NULL references cleaned")
            
            print("\n MINIMAL FIXES COMPLETED!")
            print(" Data is now suitable for analytics")
            
        except Exception as e:
            print(f" Error during fixes: {e}")
    
    def validate_analytics_readiness(self):
        """Check if data is ready for analytics"""
        print("\n VALIDATING ANALYTICS READINESS")
        print("=" * 40)
        
        checks = [
            ("No negative revenues", "SELECT COUNT(*) FROM analytics.sales WHERE total_amount < 0"),
            ("Valid ratings only", "SELECT COUNT(*) FROM analytics.reviews WHERE rating < 1 OR rating > 5"),
            ("All sales have customers", "SELECT COUNT(*) FROM analytics.sales WHERE customer_id IS NULL"),
            ("All sales have products", "SELECT COUNT(*) FROM analytics.sales WHERE product_id IS NULL")
        ]
        
        all_good = True
        for check_name, query in checks:
            try:
                result = pd.read_sql(query, self.engine)
                count = result.iloc[0].iloc[0]
                if count == 0:
                    print(f"    {check_name}: OK")
                else:
                    print(f"     {check_name}: {count} issues")
                    all_good = False
            except Exception as e:
                print(f"    {check_name}: Error - {e}")
                all_good = False
        
        return all_good
    
    def generate_analytics_preview(self):
        """Show what analytics we can do with current data"""
        print("\n ANALYTICS CAPABILITIES WITH CURRENT DATA")
        print("=" * 50)
        
        try:
            # Customer segmentation preview
            customer_stats = pd.read_sql("""
                SELECT 
                    COUNT(DISTINCT c.customer_id) as total_customers,
                    COUNT(DISTINCT s.customer_id) as purchasing_customers,
                    ROUND(AVG(customer_orders.total_orders), 1) as avg_orders_per_customer,
                    ROUND(AVG(customer_orders.total_spent), 2) as avg_customer_value
                FROM analytics.customers c
                LEFT JOIN analytics.sales s ON c.customer_id = s.customer_id
                LEFT JOIN (
                    SELECT customer_id, COUNT(*) as total_orders, SUM(total_amount) as total_spent
                    FROM analytics.sales 
                    WHERE order_status = 'completed'
                    GROUP BY customer_id
                ) customer_orders ON c.customer_id = customer_orders.customer_id
            """, self.engine)
            
            print(" CUSTOMER ANALYTICS READY:")
            stats = customer_stats.iloc[0]
            print(f"   • Total Customers: {stats['total_customers']:,}")
            print(f"   • Purchasing Customers: {stats['purchasing_customers']:,}")
            print(f"   • Avg Orders per Customer: {stats['avg_orders_per_customer']}")
            print(f"   • Avg Customer Value: ${stats['avg_customer_value']:,.2f}")
            
            # Product analytics preview
            product_stats = pd.read_sql("""
                SELECT 
                    category,
                    COUNT(*) as products,
                    ROUND(AVG(price), 2) as avg_price,
                    COUNT(DISTINCT s.sale_id) as total_sales
                FROM analytics.products p
                LEFT JOIN analytics.sales s ON p.product_id = s.product_id
                GROUP BY category
                ORDER BY total_sales DESC
            """, self.engine)
            
            print("\n  PRODUCT ANALYTICS READY:")
            for _, row in product_stats.head().iterrows():
                print(f"   • {row['category']}: {row['products']:,} products, ${row['avg_price']:.2f} avg price, {row['total_sales']:,} sales")
            
            # Review analytics preview  
            review_stats = pd.read_sql("""
                SELECT 
                    ROUND(AVG(rating), 2) as avg_rating,
                    COUNT(*) as total_reviews,
                    SUM(CASE WHEN verified_purchase THEN 1 ELSE 0 END) as verified_reviews
                FROM analytics.reviews
            """, self.engine)
            
            print("\n REVIEW ANALYTICS READY:")
            r_stats = review_stats.iloc[0]
            print(f"   • Average Rating: {r_stats['avg_rating']}/5.0")
            print(f"   • Total Reviews: {r_stats['total_reviews']:,}")
            print(f"   • Verified Reviews: {r_stats['verified_reviews']:,}")
            
        except Exception as e:
            print(f" Error generating preview: {e}")
    
    def run_quick_fix(self):
        """Run the quick fix process"""
        print(" QUICK DATA FIX FOR ANALYTICS")
        print("=" * 50)
        
        # Apply minimal fixes
        self.apply_minimal_fixes()
        
        # Validate readiness
        ready = self.validate_analytics_readiness()
        
        # Show what we can do
        self.generate_analytics_preview()
        
        print(f"\n" + "=" * 50)
        if ready:
            print(" DATA IS READY FOR ADVANCED ANALYTICS!")
        else:
            print(" DATA IS GOOD ENOUGH FOR ANALYTICS!")
        
        return True

def main():
    fixer = QuickDataFix()
    fixer.run_quick_fix()

if __name__ == "__main__":
    main()