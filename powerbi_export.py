# simple_powerbi_export.py
import pandas as pd
import numpy as np
from database_connection import DatabaseConnection
import os
from datetime import datetime

class SimplePowerBIExport:
    def __init__(self):
        """Simple, reliable Power BI data export"""
        self.db_conn = DatabaseConnection()
        self.engine = self.db_conn.get_engine()
        
    def export_customer_dashboard_data(self):
        """Export customer data with manual query execution"""
        print(" Exporting Customer Data for Power BI...")
        
        try:
            # Simple, reliable query
            with self.engine.connect() as conn:
                customers = pd.read_sql_query("""
                SELECT 
                    c.customer_id,
                    c.customer_code,
                    c.first_name || ' ' || c.last_name as customer_name,
                    c.country,
                    c.age_group,
                    c.gender,
                    c.registration_date,
                    cs.segment_name as customer_segment,
                    cs.lifetime_value as clv_amount,
                    ROUND(cs.churn_probability * 100, 1) as churn_risk_percent,
                    cs.total_orders,
                    cs.avg_order_value,
                    cs.last_purchase_date,
                    CASE 
                        WHEN cs.lifetime_value >= 5000 THEN 'Platinum'
                        WHEN cs.lifetime_value >= 2000 THEN 'Gold'
                        WHEN cs.lifetime_value >= 1000 THEN 'Silver'
                        WHEN cs.lifetime_value >= 500 THEN 'Bronze'
                        ELSE 'Basic'
                    END as clv_tier,
                    CASE 
                        WHEN cs.churn_probability >= 0.8 THEN 'Critical'
                        WHEN cs.churn_probability >= 0.6 THEN 'High'
                        WHEN cs.churn_probability >= 0.4 THEN 'Medium'
                        WHEN cs.churn_probability >= 0.2 THEN 'Low'
                        ELSE 'Very Low'
                    END as risk_category
                FROM analytics.customers c
                JOIN analytics.customer_segments cs ON c.customer_id = cs.customer_id
                WHERE cs.lifetime_value > 0
                ORDER BY cs.lifetime_value DESC
                """, conn)
            
            # Calculate additional fields in pandas (more reliable)
            customers['value_at_risk'] = (customers['clv_amount'] * 
                                        customers['churn_risk_percent'] / 100).round(2)
            
            # Action priority
            customers['action_priority'] = 'Standard Tracking'
            customers.loc[(customers['churn_risk_percent'] >= 70) & 
                         (customers['clv_amount'] >= 1000), 'action_priority'] = 'Immediate Action'
            customers.loc[(customers['churn_risk_percent'] >= 50) & 
                         (customers['clv_amount'] >= 500), 'action_priority'] = 'Proactive Outreach'
            customers.loc[customers['churn_risk_percent'] >= 30, 'action_priority'] = 'Monitor Closely'
            
            # Create output folder
            os.makedirs('dashboard_powerbi', exist_ok=True)
            customers.to_csv('dashboard_powerbi/customers_analysis.csv', index=False)
            
            print(f" Customer Data: {len(customers):,} records exported")
            return customers
            
        except Exception as e:
            print(f" Error exporting customer data: {e}")
            return pd.DataFrame()
    
    def export_sales_dashboard_data(self):
        """Export sales data for Power BI"""
        print(" Exporting Sales Data for Power BI...")
        
        try:
            with self.engine.connect() as conn:
                sales = pd.read_sql_query("""
                SELECT 
                    s.sale_id,
                    s.customer_id,
                    s.product_id,
                    s.order_date,
                    s.quantity,
                    s.unit_price,
                    s.total_amount,
                    s.discount_amount,
                    s.payment_method,
                    s.order_status,
                    p.title as product_name,
                    p.category,
                    p.subcategory,
                    p.brand,
                    p.price as product_list_price,
                    c.country as customer_country,
                    c.age_group as customer_age_group,
                    EXTRACT(YEAR FROM s.order_date) as order_year,
                    EXTRACT(MONTH FROM s.order_date) as order_month,
                    EXTRACT(QUARTER FROM s.order_date) as order_quarter
                FROM analytics.sales s
                JOIN analytics.products p ON s.product_id = p.product_id
                JOIN analytics.customers c ON s.customer_id = c.customer_id
                WHERE s.order_status = 'completed'
                ORDER BY s.order_date DESC
                LIMIT 100000
                """, conn)
            
            # Add calculated fields
            sales['net_revenue'] = sales['total_amount'] - sales['discount_amount']
            sales['discount_percentage'] = ((sales['discount_amount'] / 
                                           sales['total_amount']) * 100).round(2)
            sales['year_month'] = (sales['order_year'].astype(str) + '-' + 
                                 sales['order_month'].astype(str).str.zfill(2))
            
            sales.to_csv('dashboard_powerbi/sales_transactions.csv', index=False)
            
            print(f" Sales Data: {len(sales):,} records exported")
            return sales
            
        except Exception as e:
            print(f" Error exporting sales data: {e}")
            return pd.DataFrame()
    
    def export_product_dashboard_data(self):
        """Export product performance data"""
        print(" Exporting Product Data for Power BI...")
        
        try:
            with self.engine.connect() as conn:
                products = pd.read_sql_query("""
                SELECT 
                    p.product_id,
                    p.title as product_name,
                    p.category,
                    p.subcategory,
                    p.brand,
                    p.price as list_price,
                    p.launch_date,
                    COUNT(s.sale_id) as total_sales_count,
                    COALESCE(SUM(s.total_amount), 0) as total_revenue,
                    COALESCE(SUM(s.quantity), 0) as units_sold,
                    COALESCE(AVG(s.total_amount), 0) as avg_sale_amount,
                    COUNT(DISTINCT s.customer_id) as unique_customers,
                    COUNT(r.review_id) as review_count,
                    COALESCE(AVG(r.rating), 0) as avg_rating
                FROM analytics.products p
                LEFT JOIN analytics.sales s ON p.product_id = s.product_id 
                    AND s.order_status = 'completed'
                LEFT JOIN analytics.reviews r ON p.product_id = r.product_id
                GROUP BY p.product_id, p.title, p.category, p.subcategory, 
                         p.brand, p.price, p.launch_date
                ORDER BY total_revenue DESC
                """, conn)
            
            # Add performance categories
            products['performance_category'] = 'No Sales'
            products.loc[products['total_revenue'] > 0, 'performance_category'] = 'Low Performer'
            products.loc[products['total_revenue'] >= 10000, 'performance_category'] = 'Medium Performer'
            products.loc[products['total_revenue'] >= 50000, 'performance_category'] = 'High Performer'
            products.loc[products['total_revenue'] >= 100000, 'performance_category'] = 'Top Performer'
            
            # Rating categories
            products['rating_category'] = 'No Ratings'
            products.loc[products['avg_rating'] > 0, 'rating_category'] = 'Poor'
            products.loc[products['avg_rating'] >= 3.0, 'rating_category'] = 'Below Average'
            products.loc[products['avg_rating'] >= 3.5, 'rating_category'] = 'Average'
            products.loc[products['avg_rating'] >= 4.0, 'rating_category'] = 'Good'
            products.loc[products['avg_rating'] >= 4.5, 'rating_category'] = 'Excellent'
            
            products.to_csv('dashboard_powerbi/product_performance.csv', index=False)
            
            print(f" Product Data: {len(products):,} records exported")
            return products
            
        except Exception as e:
            print(f" Error exporting product data: {e}")
            return pd.DataFrame()
    
    def export_monthly_trends_data(self):
        """Export monthly trends data"""
        print(" Exporting Monthly Trends for Power BI...")
        
        try:
            with self.engine.connect() as conn:
                trends = pd.read_sql_query("""
                SELECT 
                    DATE_TRUNC('month', s.order_date) as month_date,
                    EXTRACT(YEAR FROM s.order_date) as year,
                    EXTRACT(MONTH FROM s.order_date) as month_number,
                    COUNT(*) as total_transactions,
                    SUM(s.total_amount) as monthly_revenue,
                    AVG(s.total_amount) as avg_order_value,
                    SUM(s.quantity) as total_units_sold,
                    COUNT(DISTINCT s.customer_id) as unique_customers,
                    SUM(CASE WHEN p.category = 'Electronics' THEN s.total_amount ELSE 0 END) as electronics_revenue,
                    SUM(CASE WHEN p.category = 'Clothing' THEN s.total_amount ELSE 0 END) as clothing_revenue,
                    SUM(CASE WHEN p.category = 'Home & Kitchen' THEN s.total_amount ELSE 0 END) as home_kitchen_revenue,
                    SUM(CASE WHEN p.category = 'Books' THEN s.total_amount ELSE 0 END) as books_revenue,
                    SUM(CASE WHEN p.category = 'Sports' THEN s.total_amount ELSE 0 END) as sports_revenue
                FROM analytics.sales s
                JOIN analytics.products p ON s.product_id = p.product_id
                WHERE s.order_status = 'completed'
                GROUP BY DATE_TRUNC('month', s.order_date),
                         EXTRACT(YEAR FROM s.order_date),
                         EXTRACT(MONTH FROM s.order_date)
                ORDER BY month_date
                """, conn)
            
            # Calculate growth rates
            trends['revenue_mom_growth'] = trends['monthly_revenue'].pct_change() * 100
            trends['customer_mom_growth'] = trends['unique_customers'].pct_change() * 100
            
            # Create year-month string for Power BI
            trends['year_month'] = (trends['year'].astype(str) + '-' + 
                                  trends['month_number'].astype(str).str.zfill(2))
            
            trends.to_csv('dashboard_powerbi/monthly_trends.csv', index=False)
            
            print(f" Monthly Trends: {len(trends):,} records exported")
            return trends
            
        except Exception as e:
            print(f" Error exporting trends data: {e}")
            return pd.DataFrame()
    
    def export_geographic_data(self):
        """Export geographic performance data"""
        print(" Exporting Geographic Data for Power BI...")
        
        try:
            with self.engine.connect() as conn:
                geographic = pd.read_sql_query("""
                SELECT 
                    c.country,
                    COUNT(DISTINCT c.customer_id) as total_customers,
                    COUNT(DISTINCT CASE WHEN s.customer_id IS NOT NULL THEN c.customer_id END) as active_customers,
                    COUNT(s.sale_id) as total_orders,
                    COALESCE(SUM(s.total_amount), 0) as total_revenue,
                    COALESCE(AVG(s.total_amount), 0) as avg_order_value,
                    COALESCE(AVG(cs.lifetime_value), 0) as avg_customer_clv,
                    COALESCE(SUM(cs.lifetime_value), 0) as total_clv
                FROM analytics.customers c
                LEFT JOIN analytics.sales s ON c.customer_id = s.customer_id 
                    AND s.order_status = 'completed'
                LEFT JOIN analytics.customer_segments cs ON c.customer_id = cs.customer_id
                GROUP BY c.country
                HAVING COUNT(DISTINCT c.customer_id) >= 100
                ORDER BY total_revenue DESC NULLS LAST
                """, conn)
            
            # Add performance categories
            geographic['market_category'] = 'Emerging Market'
            geographic.loc[geographic['total_revenue'] >= 100000, 'market_category'] = 'Growth Market'
            geographic.loc[geographic['total_revenue'] >= 500000, 'market_category'] = 'Major Market'
            geographic.loc[geographic['total_revenue'] >= 1000000, 'market_category'] = 'Top Market'
            
            # Calculate ratios
            geographic['orders_per_customer'] = (geographic['total_orders'] / 
                                               geographic['total_customers']).round(2)
            geographic['revenue_per_customer'] = (geographic['total_revenue'] / 
                                                geographic['total_customers']).round(2)
            
            geographic.to_csv('dashboard_powerbi/geographic_performance.csv', index=False)
            
            print(f" Geographic Data: {len(geographic):,} records exported")
            return geographic
            
        except Exception as e:
            print(f" Error exporting geographic data: {e}")
            return pd.DataFrame()
    
        
    def export_all_data(self):
        """Export all data with error handling"""
        print("DATA EXPORT")
        print("=" * 50)
        print("Creating reliable dashboard data...")
        print()
        
        # Export with error handling
        customers = self.export_customer_dashboard_data()
        sales = self.export_sales_dashboard_data()
        products = self.export_product_dashboard_data()
        trends = self.export_monthly_trends_data()
        geographic = self.export_geographic_data()
        
        
        # Calculate summary
        total_records = len(customers) + len(sales) + len(products) + len(trends) + len(geographic)
        successful_exports = sum(1 for df in [customers, sales, products, trends, geographic] if len(df) > 0)
        
        print(f"\n EXPORT COMPLETED!")
        print("=" * 50)
        print(f" Successful exports: {successful_exports}/5")
        print(f" Total records: {total_records:,}")
        
        if successful_exports >= 3:
            print(f"\n📁 FILES CREATED:")
            print(" customers_analysis.csv")
            print(" sales_transactions.csv")  
            print(" product_performance.csv")
            print(" monthly_trends.csv")
            print(" geographic_performance.csv")

        else:
            print(f"\n  Some exports failed, but you have enough data to proceed!")
        
        return {
            'customers': customers,
            'sales': sales,
            'products': products, 
            'trends': trends,
            'geographic': geographic,
            'total_records': total_records
        }

def main():
    """Main execution"""
    exporter = SimplePowerBIExport()
    results = exporter.export_all_data()

if __name__ == "__main__":
    main()