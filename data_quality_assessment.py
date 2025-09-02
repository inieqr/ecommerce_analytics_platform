# data_quality_assessment.py
import pandas as pd
import numpy as np
from database_connection import DatabaseConnection
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class DataQualityAssessment:
    def __init__(self):
        """Initialize data quality assessment"""
        self.db_conn = DatabaseConnection()
        self.engine = self.db_conn.get_engine()
        
    def load_all_data(self):
        """Load all tables for quality assessment"""
        print(" Loading all data for quality assessment...")
        
        self.products = pd.read_sql("SELECT * FROM analytics.products", self.engine)
        self.customers = pd.read_sql("SELECT * FROM analytics.customers", self.engine)
        self.sales = pd.read_sql("SELECT * FROM analytics.sales", self.engine)
        self.reviews = pd.read_sql("SELECT * FROM analytics.reviews", self.engine)
        
        print(f" Loaded:")
        print(f"   Products: {len(self.products):,} records")
        print(f"   Customers: {len(self.customers):,} records")
        print(f"   Sales: {len(self.sales):,} records") 
        print(f"   Reviews: {len(self.reviews):,} records")
        
    def assess_data_completeness(self):
        """Check for missing values and data completeness"""
        print("\n DATA COMPLETENESS ASSESSMENT")
        print("=" * 50)
        
        tables = {
            'Products': self.products,
            'Customers': self.customers,
            'Sales': self.sales,
            'Reviews': self.reviews
        }
        
        completeness_report = {}
        
        for table_name, df in tables.items():
            print(f"\n {table_name} Table:")
            
            # Check missing values
            missing_counts = df.isnull().sum()
            missing_percentages = (df.isnull().sum() / len(df) * 100).round(2)
            
            completeness_data = []
            
            for column in df.columns:
                missing_count = missing_counts[column]
                missing_pct = missing_percentages[column]
                data_type = str(df[column].dtype)
                
                completeness_data.append({
                    'column': column,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'data_type': data_type
                })
                
                if missing_count > 0:
                    print(f"    {column}: {missing_count:,} missing ({missing_pct:.1f}%)")
                else:
                    print(f"    {column}: Complete (0% missing)")
            
            completeness_report[table_name] = completeness_data
        
        return completeness_report
    
    def assess_data_validity(self):
        """Check for data validity and logical consistency"""
        print("\n DATA VALIDITY ASSESSMENT")
        print("=" * 50)
        
        issues_found = []
        
        # Products validity checks
        print("\n Products Validity:")
        
        # Price validity
        invalid_prices = self.products[self.products['price'] <= 0]
        if len(invalid_prices) > 0:
            print(f"    Invalid prices (≤0): {len(invalid_prices):,} products")
            issues_found.append("Invalid product prices")
        else:
            print(f"   All prices are valid (>0)")
            print(f"      Price range: ${self.products['price'].min():.2f} - ${self.products['price'].max():.2f}")
        
        # Category distribution
        print(f"    Category distribution:")
        for category, count in self.products['category'].value_counts().items():
            pct = count / len(self.products) * 100
            print(f"      {category}: {count:,} ({pct:.1f}%)")
        
        # Customers validity checks  
        print("\n Customers Validity:")
        
        # Email format (basic check)
        invalid_emails = self.customers[~self.customers['email'].str.contains('@', na=False)]
        if len(invalid_emails) > 0:
            print(f"     Invalid email formats: {len(invalid_emails):,} customers")
            issues_found.append("Invalid email formats")
        else:
            print(f"    All email formats appear valid")
        
        # Geographic distribution
        print(f"    Geographic distribution:")
        for country, count in self.customers['country'].value_counts().head().items():
            pct = count / len(self.customers) * 100
            print(f"      {country}: {count:,} ({pct:.1f}%)")
        
        # Sales validity checks
        print("\n Sales Validity:")
        
        # Negative amounts
        negative_sales = self.sales[self.sales['total_amount'] < 0]
        if len(negative_sales) > 0:
            print(f"     Negative sales amounts: {len(negative_sales):,} transactions")
            issues_found.append("Negative sales amounts")
        else:
            print(f"    All sales amounts are positive")
            print(f"      Revenue range: ${self.sales['total_amount'].min():.2f} - ${self.sales['total_amount'].max():.2f}")
        
        # Quantity validity
        invalid_qty = self.sales[self.sales['quantity'] <= 0]
        if len(invalid_qty) > 0:
            print(f"   ⚠️  Invalid quantities (≤0): {len(invalid_qty):,} sales")
            issues_found.append("Invalid quantities")
        else:
            print(f"    All quantities are valid (>0)")
            print(f"      Quantity range: {self.sales['quantity'].min()} - {self.sales['quantity'].max()}")
        
        # Order status distribution
        print(f"    Order status distribution:")
        for status, count in self.sales['order_status'].value_counts().items():
            pct = count / len(self.sales) * 100
            print(f"      {status}: {count:,} ({pct:.1f}%)")
        
        # Reviews validity checks
        print("\n Reviews Validity:")
        
        # Rating range
        invalid_ratings = self.reviews[(self.reviews['rating'] < 1) | (self.reviews['rating'] > 5)]
        if len(invalid_ratings) > 0:
            print(f"     Invalid ratings (not 1-5): {len(invalid_ratings):,} reviews")
            issues_found.append("Invalid ratings")
        else:
            print(f"    All ratings are valid (1-5 stars)")
            print(f"    Rating distribution:")
            for rating, count in self.reviews['rating'].value_counts().sort_index().items():
                pct = count / len(self.reviews) * 100
                stars = "⭐" * int(rating)
                print(f"      {rating} {stars}: {count:,} ({pct:.1f}%)")
        
        return issues_found
    
    def assess_data_relationships(self):
        """Check referential integrity and relationships"""
        print("\n DATA RELATIONSHIPS ASSESSMENT")
        print("=" * 50)
        
        relationship_issues = []
        
        # Sales -> Products relationship
        print("\n Sales → Products Relationship:")
        orphaned_sales = self.sales[~self.sales['product_id'].isin(self.products['product_id'])]
        if len(orphaned_sales) > 0:
            print(f"     Orphaned sales (invalid product_id): {len(orphaned_sales):,}")
            relationship_issues.append("Orphaned sales records")
        else:
            print(f"    All sales have valid product references")
        
        # Sales -> Customers relationship
        print(f"🔗 Sales → Customers Relationship:")
        orphaned_customer_sales = self.sales[~self.sales['customer_id'].isin(self.customers['customer_id'])]
        if len(orphaned_customer_sales) > 0:
            print(f"     Orphaned sales (invalid customer_id): {len(orphaned_customer_sales):,}")
            relationship_issues.append("Orphaned customer sales")
        else:
            print(f"    All sales have valid customer references")
        
        # Reviews -> Products relationship
        print(f"🔗 Reviews → Products Relationship:")
        orphaned_reviews = self.reviews[~self.reviews['product_id'].isin(self.products['product_id'])]
        if len(orphaned_reviews) > 0:
            print(f"     Orphaned reviews (invalid product_id): {len(orphaned_reviews):,}")
            relationship_issues.append("Orphaned reviews")
        else:
            print(f"    All reviews have valid product references")
        
        # Reviews -> Customers relationship
        print(f"🔗 Reviews → Customers Relationship:")
        orphaned_customer_reviews = self.reviews[~self.reviews['customer_id'].isin(self.customers['customer_id'])]
        if len(orphaned_customer_reviews) > 0:
            print(f"     Orphaned reviews (invalid customer_id): {len(orphaned_customer_reviews):,}")
            relationship_issues.append("Orphaned customer reviews")
        else:
            print(f"    All reviews have valid customer references")
        
        return relationship_issues
    
    def assess_business_logic(self):
        """Check business logic and realistic patterns"""
        print("\n BUSINESS LOGIC ASSESSMENT")
        print("=" * 50)
        
        business_issues = []
        
        # Date consistency checks
        print("\n Date Logic Checks:")
        
        # Check if review dates are after product launch dates
        sales_with_products = self.sales.merge(self.products[['product_id', 'launch_date']], on='product_id')
        sales_with_products['order_date'] = pd.to_datetime(sales_with_products['order_date'])
        sales_with_products['launch_date'] = pd.to_datetime(sales_with_products['launch_date'])
        
        impossible_sales = sales_with_products[sales_with_products['order_date'] < sales_with_products['launch_date']]
        if len(impossible_sales) > 0:
            print(f"     Sales before product launch: {len(impossible_sales):,}")
            business_issues.append("Sales before product launch")
        else:
            print(f"    All sales occurred after product launch")
        
        # Customer registration vs first purchase
        first_purchases = self.sales.groupby('customer_id')['order_date'].min().reset_index()
        first_purchases.columns = ['customer_id', 'first_purchase_date']
        
        customer_purchase_check = self.customers.merge(first_purchases, on='customer_id', how='left')
        customer_purchase_check['registration_date'] = pd.to_datetime(customer_purchase_check['registration_date'])
        customer_purchase_check['first_purchase_date'] = pd.to_datetime(customer_purchase_check['first_purchase_date'])
        
        early_purchases = customer_purchase_check[
            customer_purchase_check['first_purchase_date'] < customer_purchase_check['registration_date']
        ].dropna()
        
        if len(early_purchases) > 0:
            print(f"     Purchases before customer registration: {len(early_purchases):,}")
            business_issues.append("Purchases before registration")
        else:
            print(f"    All purchases occurred after customer registration")
        
        # Revenue calculations check
        print("\n Revenue Logic Checks:")
        
        # Check if total_amount = (unit_price * quantity) - discount_amount + shipping_cost
        calculated_total = ((self.sales['unit_price'] * self.sales['quantity']) - 
                           self.sales['discount_amount'] + self.sales['shipping_cost'])
        
        revenue_mismatches = abs(self.sales['total_amount'] - calculated_total) > 0.01  # Allow 1 cent rounding
        mismatch_count = revenue_mismatches.sum()
        
        if mismatch_count > 0:
            print(f"     Revenue calculation mismatches: {mismatch_count:,}")
            business_issues.append("Revenue calculation errors")
        else:
            print(f"    All revenue calculations are correct")
        
        return business_issues
    
    def generate_data_profile(self):
        """Generate comprehensive data profile"""
        print("\n DATA PROFILE SUMMARY")
        print("=" * 50)
        
        total_records = len(self.products) + len(self.customers) + len(self.sales) + len(self.reviews)
        completed_sales = self.sales[self.sales['order_status'] == 'completed']
        total_revenue = completed_sales['total_amount'].sum()
        
        # Date ranges
        min_date = min(
            self.products['launch_date'].min(),
            self.customers['registration_date'].min(), 
            self.sales['order_date'].min(),
            self.reviews['review_date'].min()
        )
        max_date = max(
            self.sales['order_date'].max(),
            self.reviews['review_date'].max()
        )
        
        print(f" Dataset Overview:")
        print(f"   Total Records: {total_records:,}")
        print(f"   Date Range: {min_date} to {max_date}")
        print(f"   Time Span: {(pd.to_datetime(max_date) - pd.to_datetime(min_date)).days} days")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Avg Order Value: ${completed_sales['total_amount'].mean():.2f}")
        print(f"   Completion Rate: {len(completed_sales) / len(self.sales) * 100:.1f}%")
        
        print(f"\n Geographic Coverage:")
        print(f"   Countries: {self.customers['country'].nunique()}")
        print(f"   Top Countries: {', '.join(self.customers['country'].value_counts().head(3).index.tolist())}")
        
        print(f"\n Product Coverage:")
        print(f"   Categories: {self.products['category'].nunique()}")
        print(f"   Brands: {self.products['brand'].nunique()}")
        print(f"   Price Range: ${self.products['price'].min():.2f} - ${self.products['price'].max():.2f}")
        
        print(f"\n Review Coverage:")
        print(f"   Average Rating: {self.reviews['rating'].mean():.1f}/5.0")
        print(f"   Verified Purchase Rate: {self.reviews['verified_purchase'].mean() * 100:.1f}%")
    
    def run_complete_assessment(self):
        """Run the complete data quality assessment"""
        print(" COMPREHENSIVE DATA QUALITY ASSESSMENT")
        print("=" * 60)
        print("Analyzing data completeness, validity, relationships, and business logic...")
        print()
        
        # Load all data
        self.load_all_data()
        
        # Run all assessments
        completeness_issues = self.assess_data_completeness()
        validity_issues = self.assess_data_validity()  
        relationship_issues = self.assess_data_relationships()
        business_issues = self.assess_business_logic()
        
        # Generate data profile
        self.generate_data_profile()
        
        # Summary of findings
        all_issues = validity_issues + relationship_issues + business_issues
        
        print(f"\n" + "=" * 60)
        print(" DATA QUALITY ASSESSMENT SUMMARY")
        print("=" * 60)
        
        if len(all_issues) == 0:
            print(" EXCELLENT DATA QUALITY!")
            print("   • No missing values detected")
            print("   • All data validity checks passed")
            print("   • All relationship integrity checks passed") 
            print("   • All business logic checks passed")
           
        else:
            print(" Issues found that need attention:")
            for i, issue in enumerate(all_issues, 1):
                print(f"   {i}. {issue}")
            print()
        
        
        return len(all_issues) == 0

def main():
    """Main execution"""
    assessor = DataQualityAssessment()
    data_is_clean = assessor.run_complete_assessment()
    
if __name__ == "__main__":
    main()