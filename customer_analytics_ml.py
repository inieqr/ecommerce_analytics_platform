# customer_analytics_ml_fixed.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from database_connection import DatabaseConnection
from datetime import datetime, timedelta
from sqlalchemy import text
import warnings
import uuid
warnings.filterwarnings('ignore')

class CustomerAnalyticsML:
    def __init__(self):
        """Initialize Customer Analytics & ML Pipeline"""
        self.db_conn = DatabaseConnection()
        self.engine = self.db_conn.get_engine()
        self.insights = {}
        
    def load_customer_data(self):
        """Load comprehensive customer data for analytics"""
        print("Loading customer data for advanced analytics...")
        
        # Comprehensive customer query with business metrics
        customer_query = """
        SELECT 
            c.customer_id,
            c.customer_code,
            c.first_name,
            c.last_name,
            c.email,
            c.country,
            c.age_group,
            c.gender,
            c.registration_date,
            
            -- Sales metrics
            COALESCE(sales_stats.total_orders, 0) as total_orders,
            COALESCE(sales_stats.total_revenue, 0) as total_revenue,
            COALESCE(sales_stats.avg_order_value, 0) as avg_order_value,
            COALESCE(sales_stats.first_order_date, c.registration_date) as first_order_date,
            COALESCE(sales_stats.last_order_date, c.registration_date) as last_order_date,
            COALESCE(sales_stats.days_since_last_order, 9999) as days_since_last_order,
            
            -- Review metrics  
            COALESCE(review_stats.total_reviews, 0) as total_reviews,
            COALESCE(review_stats.avg_rating_given, 0) as avg_rating_given,
            
            -- Product diversity
            COALESCE(product_stats.categories_purchased, 0) as categories_purchased,
            COALESCE(product_stats.brands_purchased, 0) as brands_purchased
            
        FROM analytics.customers c
        
        LEFT JOIN (
            SELECT 
                customer_id,
                COUNT(*) as total_orders,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_order_value,
                MIN(order_date) as first_order_date,
                MAX(order_date) as last_order_date,
                CURRENT_DATE - MAX(order_date) as days_since_last_order
            FROM analytics.sales 
            WHERE order_status = 'completed'
            GROUP BY customer_id
        ) sales_stats ON c.customer_id = sales_stats.customer_id
        
        LEFT JOIN (
            SELECT 
                customer_id,
                COUNT(*) as total_reviews,
                AVG(rating) as avg_rating_given
            FROM analytics.reviews
            GROUP BY customer_id
        ) review_stats ON c.customer_id = review_stats.customer_id
        
        LEFT JOIN (
            SELECT 
                s.customer_id,
                COUNT(DISTINCT p.category) as categories_purchased,
                COUNT(DISTINCT p.brand) as brands_purchased
            FROM analytics.sales s
            JOIN analytics.products p ON s.product_id = p.product_id
            WHERE s.order_status = 'completed'
            GROUP BY s.customer_id
        ) product_stats ON c.customer_id = product_stats.customer_id
        """
        
        self.customer_data = pd.read_sql(customer_query, self.engine)
        print(f"Loaded {len(self.customer_data):,} customers with comprehensive metrics")
        
        # Calculate additional business metrics
        self.customer_data['customer_lifetime_days'] = (
            pd.to_datetime(self.customer_data['last_order_date']) - 
            pd.to_datetime(self.customer_data['first_order_date'])
        ).dt.days + 1
        
        self.customer_data['purchase_frequency'] = (
            self.customer_data['total_orders'] / 
            (self.customer_data['customer_lifetime_days'] / 30)
        ).fillna(0)
        
        return self.customer_data
    
    def create_robust_rfm_scores(self, data, column, ascending=True):
        """Create robust RFM scores that handle duplicate values"""
        unique_values = data[column].nunique()
        
        if unique_values <= 5:
            # Use simple ranking for limited unique values
            if ascending:
                scores = data[column].rank(method='dense', ascending=True)
                scores = np.ceil(scores / scores.max() * 5).astype(int)
            else:
                scores = data[column].rank(method='dense', ascending=False) 
                scores = np.ceil(scores / scores.max() * 5).astype(int)
        else:
            # Use quantile-based scoring for sufficient unique values
            try:
                if ascending:
                    scores = pd.qcut(data[column], q=5, labels=[1,2,3,4,5], duplicates='drop')
                else:
                    scores = pd.qcut(data[column], q=5, labels=[5,4,3,2,1], duplicates='drop')
                scores = scores.astype(int)
            except ValueError:
                # Fallback to percentile-based scoring
                percentiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                if ascending:
                    scores = pd.cut(data[column], 
                                  bins=data[column].quantile(percentiles).drop_duplicates(),
                                  labels=range(1, len(data[column].quantile(percentiles).drop_duplicates())),
                                  include_lowest=True, duplicates='drop')
                else:
                    bins = data[column].quantile(percentiles).drop_duplicates()
                    labels = list(range(len(bins)-1, 0, -1))
                    scores = pd.cut(data[column], bins=bins, labels=labels,
                                  include_lowest=True, duplicates='drop')
                scores = scores.astype(int)
        
        return scores
    
    def perform_rfm_analysis(self):
        """Perform robust RFM (Recency, Frequency, Monetary) analysis"""
        print("\n🔍 Performing RFM Analysis...")
        
        # Filter customers who have made purchases
        purchasing_customers = self.customer_data[self.customer_data['total_orders'] > 0].copy()
        
        if len(purchasing_customers) == 0:
            print("No purchasing customers found")
            return None
        
        print(f"Analyzing {len(purchasing_customers):,} purchasing customers")
        
        # Calculate robust RFM scores
        print("Calculating RFM scores...")
        
        # Recency: Lower days since last order = higher score (recent = good)
        purchasing_customers['R_Score'] = self.create_robust_rfm_scores(
            purchasing_customers, 'days_since_last_order', ascending=False
        )
        
        # Frequency: Higher total orders = higher score  
        purchasing_customers['F_Score'] = self.create_robust_rfm_scores(
            purchasing_customers, 'total_orders', ascending=True
        )
        
        # Monetary: Higher total revenue = higher score
        purchasing_customers['M_Score'] = self.create_robust_rfm_scores(
            purchasing_customers, 'total_revenue', ascending=True
        )
        
        print(f"RFM scores calculated successfully")
        
        # Create RFM segments based on scores
        def categorize_customer(row):
            """Categorize customers based on RFM scores with flexible thresholds"""
            try:
                r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            except (ValueError, TypeError):
                return 'Unclassified'
            
            # VIP Champions: High scores across all dimensions
            if r >= 4 and f >= 4 and m >= 4:
                return 'VIP Champions'
            # Loyal Customers: High frequency and monetary
            elif f >= 4 and m >= 3:
                return 'Loyal Customers'
            # Big Spenders: High monetary value
            elif m >= 4:
                return 'Big Spenders'
            # At Risk: High value but not recent
            elif m >= 3 and r <= 2:
                return 'At Risk'
            # New Customers: Recent but low frequency
            elif r >= 4 and f <= 2:
                return 'New Customers'
            # Promising: Good recent activity
            elif r >= 3 and f >= 3:
                return 'Promising'
            else:
                return 'Standard Customers'
        
        purchasing_customers['RFM_Segment'] = purchasing_customers.apply(categorize_customer, axis=1)
        
        # Generate RFM insights
        rfm_summary = purchasing_customers.groupby('RFM_Segment').agg({
            'customer_id': 'count',
            'total_revenue': ['mean', 'sum'],
            'total_orders': 'mean',
            'days_since_last_order': 'mean',
            'avg_order_value': 'mean'
        }).round(2)
        
        rfm_summary.columns = ['customer_count', 'avg_revenue', 'total_segment_revenue', 
                              'avg_orders', 'avg_days_since_last', 'avg_order_value']
        
        self.rfm_data = purchasing_customers
        self.rfm_summary = rfm_summary
        
        print(f" RFM Analysis complete:")
        total_customers = len(purchasing_customers)
        total_revenue = purchasing_customers['total_revenue'].sum()
        
        for segment, data in rfm_summary.iterrows():
            pct = (data['customer_count'] / total_customers) * 100
            revenue_pct = (data['total_segment_revenue'] / total_revenue) * 100
            print(f"   • {segment}: {data['customer_count']:,} customers ({pct:.1f}%)")
            print(f"     Revenue: ${data['total_segment_revenue']:,.2f} ({revenue_pct:.1f}% of total)")
            print(f"     Avg CLV: ${data['avg_revenue']:,.2f}, Avg Orders: {data['avg_orders']:.1f}")
        
        return rfm_summary
    
    def build_customer_segmentation_model(self):
        """Build ML-powered customer segmentation using K-means clustering"""
        print("\n Building ML Customer Segmentation Model...")
        
        # Prepare features for clustering
        feature_columns = [
            'total_orders', 'total_revenue', 'avg_order_value', 
            'days_since_last_order', 'categories_purchased', 
            'brands_purchased', 'purchase_frequency'
        ]
        
        # Filter customers with purchases for meaningful segmentation
        active_customers = self.customer_data[self.customer_data['total_orders'] > 0].copy()
        
        if len(active_customers) < 100:
            print(" Insufficient active customers for ML segmentation")
            return None
        
        print(f" Training on {len(active_customers):,} active customers")
        
        # Prepare and clean features
        X = active_customers[feature_columns].fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use optimal number of clusters (5 for business interpretability)
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        active_customers['ML_Cluster'] = cluster_labels
        
        # Analyze cluster characteristics
        cluster_analysis = active_customers.groupby('ML_Cluster').agg({
            'customer_id': 'count',
            'total_revenue': ['mean', 'sum'],
            'total_orders': 'mean',
            'avg_order_value': 'mean',
            'days_since_last_order': 'mean',
            'categories_purchased': 'mean',
            'purchase_frequency': 'mean'
        }).round(2)
        
        # Create business-friendly cluster names based on characteristics
        cluster_profiles = {}
        for cluster_id in range(optimal_k):
            cluster_data = cluster_analysis.loc[cluster_id]
            avg_revenue = cluster_data[('total_revenue', 'mean')]
            avg_orders = cluster_data[('total_orders', 'mean')]
            avg_recency = cluster_data[('days_since_last_order', 'mean')]
            
            if avg_revenue >= 3000:
                name = 'Premium Customers'
            elif avg_revenue >= 1500 and avg_orders >= 5:
                name = 'Loyal Regulars'
            elif avg_recency <= 60:
                name = 'Recent Shoppers'
            elif avg_orders >= 3:
                name = 'Frequent Buyers'
            else:
                name = 'Occasional Customers'
                
            cluster_profiles[cluster_id] = name
        
        active_customers['ML_Segment_Name'] = active_customers['ML_Cluster'].map(cluster_profiles)
        
        self.ml_segments = active_customers
        self.ml_cluster_analysis = cluster_analysis
        self.segmentation_model = kmeans
        self.feature_scaler = scaler
        
        print(f" ML Segmentation complete with {optimal_k} clusters:")
        total_revenue = active_customers['total_revenue'].sum()
        for cluster_id in range(optimal_k):
            name = cluster_profiles[cluster_id]
            count = (active_customers['ML_Cluster'] == cluster_id).sum()
            avg_revenue = cluster_analysis.loc[cluster_id, ('total_revenue', 'mean')]
            segment_revenue = cluster_analysis.loc[cluster_id, ('total_revenue', 'sum')]
            revenue_pct = (segment_revenue / total_revenue) * 100
            print(f"   • {name}: {count:,} customers, ${avg_revenue:,.2f} avg revenue ({revenue_pct:.1f}% of total)")
        
        return cluster_analysis
    
    def build_churn_prediction_model(self):
        """Build ML model to predict customer churn"""
        print("\n Building Churn Prediction Model...")
        
        # Define churn based on days since last order
        active_customers = self.customer_data[self.customer_data['total_orders'] > 0].copy()
        
        if len(active_customers) < 100:
            print(" Insufficient data for churn modeling")
            return None
        
        # Calculate reasonable churn threshold
        median_recency = active_customers['days_since_last_order'].median()
        q75_recency = active_customers['days_since_last_order'].quantile(0.75)
        churn_threshold = max(180, q75_recency)  # At least 180 days or 75th percentile
        
        active_customers['is_churned'] = (active_customers['days_since_last_order'] > churn_threshold).astype(int)
        
        churn_rate = active_customers['is_churned'].mean()
        print(f" Churn threshold: {churn_threshold:.0f} days")
        print(f" Overall churn rate: {churn_rate:.1%}")
        
        if churn_rate < 0.05 or churn_rate > 0.95:
            print(" Extreme churn rate detected - model may have limited predictive power")
        
        # Features for churn prediction
        churn_features = [
            'total_orders', 'total_revenue', 'avg_order_value',
            'days_since_last_order', 'categories_purchased', 
            'brands_purchased', 'purchase_frequency', 'total_reviews'
        ]
        
        X = active_customers[churn_features].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        y = active_customers['is_churned']
        
        # Split data
        if len(X) < 50:
            print(" Small dataset - using simple train/test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': churn_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Add churn probabilities to all active customers
        X_all = active_customers[churn_features].fillna(0).replace([np.inf, -np.inf], 0)
        active_customers['churn_probability'] = rf_model.predict_proba(X_all)[:, 1]
        
        # Identify high-risk customers (high churn probability + high value)
        high_risk_threshold = min(0.7, active_customers['churn_probability'].quantile(0.9))
        value_threshold = active_customers['total_revenue'].quantile(0.7)
        
        high_risk_customers = active_customers[
            (active_customers['churn_probability'] > high_risk_threshold) & 
            (active_customers['total_revenue'] > value_threshold)
        ]
        
        self.churn_model = rf_model
        self.churn_features = churn_features
        self.churn_feature_importance = feature_importance
        self.high_risk_customers = high_risk_customers
        
        print(f"Churn Model Performance:")
        print(f"   • Accuracy: {accuracy:.1%}")
        print(f"   • High-risk valuable customers: {len(high_risk_customers):,}")
        if len(high_risk_customers) > 0:
            print(f"   • Potential retention value: ${high_risk_customers['total_revenue'].sum():,.2f}")
        
        print(f" Top predictive features:")
        for _, row in feature_importance.head(3).iterrows():
            print(f"      • {row['feature']}: {row['importance']:.3f}")
        
        return accuracy, feature_importance
    
    def calculate_customer_lifetime_value(self):
        """Calculate and predict Customer Lifetime Value (CLV)"""
        print("\n Calculating Customer Lifetime Value...")
        
        active_customers = self.customer_data[self.customer_data['total_orders'] > 0].copy()
        
        if len(active_customers) == 0:
            print(" No active customers for CLV calculation")
            return None
        
        # Historical CLV (what they've already spent)
        active_customers['historical_clv'] = active_customers['total_revenue']
        
        # Predicted CLV using business metrics
        active_customers['predicted_monthly_orders'] = np.maximum(
            active_customers['purchase_frequency'], 0.1
        )
        
        # Predict future lifetime based on current behavior and churn risk
        if hasattr(self, 'high_risk_customers') and len(self.high_risk_customers) > 0:
            # Use churn probability if available
            active_customers['predicted_lifetime_months'] = np.where(
                active_customers.get('churn_probability', 0) > 0.7, 6,  # High churn risk
                np.where(active_customers['days_since_last_order'] < 90, 24, 12)  # Based on recency
            )
        else:
            # Simple recency-based prediction
            active_customers['predicted_lifetime_months'] = np.where(
                active_customers['days_since_last_order'] < 90, 24,
                np.where(active_customers['days_since_last_order'] < 180, 12, 6)
            )
        
        # Calculate predicted CLV
        active_customers['predicted_clv'] = (
            active_customers['avg_order_value'] * 
            active_customers['predicted_monthly_orders'] * 
            active_customers['predicted_lifetime_months']
        ).round(2)
        
        # Combined CLV (historical + predicted)
        active_customers['total_clv'] = (
            active_customers['historical_clv'] + 
            active_customers['predicted_clv']
        ).round(2)
        
        # CLV segments
        def clv_segment(clv):
            if clv >= 5000:
                return 'Platinum'
            elif clv >= 2000:
                return 'Gold'  
            elif clv >= 1000:
                return 'Silver'
            elif clv >= 500:
                return 'Bronze'
            else:
                return 'Basic'
        
        active_customers['clv_segment'] = active_customers['total_clv'].apply(clv_segment)
        
        # CLV insights
        clv_summary = active_customers.groupby('clv_segment').agg({
            'customer_id': 'count',
            'total_clv': ['mean', 'sum'],
            'historical_clv': 'mean',
            'predicted_clv': 'mean'
        }).round(2)
        
        self.clv_data = active_customers
        self.clv_summary = clv_summary
        
        print(f"CLV Analysis complete:")
        total_clv = active_customers['total_clv'].sum()
        for segment in ['Platinum', 'Gold', 'Silver', 'Bronze', 'Basic']:
            if segment in clv_summary.index:
                data = clv_summary.loc[segment]
                count = data[('customer_id', 'count')]
                avg_clv = data[('total_clv', 'mean')]
                segment_value = data[('total_clv', 'sum')]
                pct_value = (segment_value / total_clv) * 100
                print(f"   • {segment}: {count:,} customers, ${avg_clv:,.2f} avg CLV ({pct_value:.1f}% of total value)")
        
        return clv_summary
    
    def generate_business_insights(self):
        """Generate comprehensive business insights from all analyses"""
        print("\n GENERATING BUSINESS INSIGHTS")
        print("=" * 50)
        
        insights = {}
        
        # Customer base insights
        total_customers = len(self.customer_data)
        active_customers = len(self.customer_data[self.customer_data['total_orders'] > 0])
        
        insights['customer_base'] = {
            'total_customers': total_customers,
            'active_customers': active_customers,
            'activation_rate': f"{(active_customers/total_customers)*100:.1f}%"
        }
        
        print(f"CUSTOMER BASE:")
        print(f"   • Total Customers: {total_customers:,}")
        print(f"   • Active Customers: {active_customers:,}")
        print(f"   • Activation Rate: {insights['customer_base']['activation_rate']}")
        
        # Revenue insights
        if hasattr(self, 'clv_data') and len(self.clv_data) > 0:
            total_historical_revenue = self.clv_data['historical_clv'].sum()
            total_predicted_revenue = self.clv_data['predicted_clv'].sum()
            
            insights['revenue'] = {
                'historical_revenue': total_historical_revenue,
                'predicted_revenue': total_predicted_revenue,
                'total_opportunity': total_historical_revenue + total_predicted_revenue
            }
            
            print(f"\n REVENUE INSIGHTS:")
            print(f"   • Historical Revenue: ${total_historical_revenue:,.2f}")
            print(f"   • Predicted Future Revenue: ${total_predicted_revenue:,.2f}")
            print(f"   • Total Customer Value: ${insights['revenue']['total_opportunity']:,.2f}")
        
        # High-value opportunities
        if hasattr(self, 'high_risk_customers') and len(self.high_risk_customers) > 0:
            retention_value = self.high_risk_customers['total_revenue'].sum()
            insights['retention_opportunity'] = {
                'at_risk_customers': len(self.high_risk_customers),
                'retention_value': retention_value
            }
            
            print(f"\n RETENTION OPPORTUNITIES:")
            print(f"   • High-Risk Valuable Customers: {len(self.high_risk_customers):,}")
            print(f"   • Retention Value at Risk: ${retention_value:,.2f}")
        
        # Segmentation insights
        if hasattr(self, 'rfm_summary') and len(self.rfm_summary) > 0:
            top_segment = self.rfm_summary.loc[self.rfm_summary['total_segment_revenue'].idxmax()]
            insights['top_segment'] = {
                'name': self.rfm_summary['total_segment_revenue'].idxmax(),
                'revenue': top_segment['total_segment_revenue'],
                'customers': top_segment['customer_count']
            }
            
            print(f"\n TOP PERFORMING SEGMENT:")
            segment_name = insights['top_segment']['name']
            print(f"   • Segment: {segment_name}")
            print(f"   • Customers: {insights['top_segment']['customers']:,}")
            print(f"   • Revenue: ${insights['top_segment']['revenue']:,.2f}")
        
        self.insights = insights
        return insights
    
    def save_results_to_database(self):
        """Save analytics results back to database for dashboard use"""
        print("\n Saving results to database...")
        
        try:
            # Save customer segments if available
            if hasattr(self, 'ml_segments') and len(self.ml_segments) > 0:
                segments_for_db = self.ml_segments[[
                    'customer_id', 'ML_Segment_Name', 'total_revenue',
                    'total_orders', 'avg_order_value', 'days_since_last_order'
                ]].copy()
                
                # Add required fields for database
                segments_for_db['segment_id'] = [str(uuid.uuid4()) for _ in range(len(segments_for_db))]
                segments_for_db['segment_score'] = self.ml_segments['ML_Cluster'] + 1
                segments_for_db['lifetime_value'] = self.ml_segments.get('total_clv', self.ml_segments['total_revenue'])
                segments_for_db['acquisition_cost'] = 50.0  # Business assumption
                segments_for_db['churn_probability'] = self.ml_segments.get('churn_probability', 0.0)
                segments_for_db['last_purchase_date'] = pd.to_datetime(self.ml_segments['last_order_date']).dt.date
                
                # Rename columns to match database schema
                segments_for_db = segments_for_db.rename(columns={
                    'ML_Segment_Name': 'segment_name'
                })
                
                # Clear existing segments and insert new ones
                with self.engine.begin() as conn:
                    conn.execute(text("TRUNCATE TABLE analytics.customer_segments"))

               # with self.engine.connect() as conn:
               #     conn.execute("TRUNCATE TABLE analytics.customer_segments")
               #     conn.commit()
                
                segments_for_db.to_sql('customer_segments', self.engine, schema='analytics',
                                      if_exists='append', index=False)
                
                print(f" Saved customer segments: {len(segments_for_db):,} records")
                
        except Exception as e:
            print(f"  Error saving segments: {e}")
        
        # Save insights summary
        try:
            import json
            with open('customer_analytics_insights.json', 'w') as f:
                json.dump(self.insights, f, indent=2, default=str)
            print(f"   Saved insights summary to JSON file")
        except Exception as e:
            print(f"   Error saving insights: {e}")
    
    def run_complete_analytics(self):
        """Run the complete customer analytics and ML pipeline"""
        print("CUSTOMER ANALYTICS & MACHINE LEARNING PIPELINE")
        print("=" * 60)
        print("Building advanced customer intelligence and predictive models...")
        print()
        
        try:
            # Load data
            self.load_customer_data()
            
            # Perform analyses
            rfm_results = self.perform_rfm_analysis()
            ml_results = self.build_customer_segmentation_model()
            churn_results = self.build_churn_prediction_model()
            clv_results = self.calculate_customer_lifetime_value()
            
            # Generate insights
            insights = self.generate_business_insights()
            
            # Save results
            self.save_results_to_database()
            
            print(f"\n" + "=" * 60)
            print(" CUSTOMER ANALYTICS PIPELINE COMPLETED!")
            print("=" * 60)
            
            print(f"\n PORTFOLIO ACHIEVEMENTS:")
            print(f"Advanced RFM customer segmentation")
            print(f"Machine learning clustering model")
            if churn_results:
                print(f" Churn prediction with {churn_results[0]:.1%} accuracy")
            else:
                print(f" Churn prediction model developed")
            print(f" Customer lifetime value modeling")
            print(f" Comprehensive business insights generation")
            print(f" Production-ready data pipeline")
            
            print(f"\n BUSINESS IMPACT:")
            if 'revenue' in insights:
                print(f" Total customer value quantified: ${insights['revenue']['total_opportunity']:,.2f}")
            if 'retention_opportunity' in insights:
                print(f" Retention opportunities identified: ${insights['retention_opportunity']['retention_value']:,.2f}")
            print(f"Customer segments ready for targeted campaigns")
            print(f"Predictive models ready for production deployment")
            
            return {
                'rfm_analysis': rfm_results,
                'ml_segmentation': ml_results, 
                'churn_prediction': churn_results,
                'clv_analysis': clv_results,
                'business_insights': insights
            }
            
        except Exception as e:
            print(f"\n Error in analytics pipeline: {e}")
            print("Continuing with available results...")
            return None

def main():
    """Main execution"""
    analytics = CustomerAnalyticsML()
    results = analytics.run_complete_analytics()

if __name__ == "__main__":
    main()