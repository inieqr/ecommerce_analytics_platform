# generate_sample_data.py
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random
from faker import Faker
import os
from database_connection import DatabaseConnection

# Initialize faker for realistic data
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class EcommerceDataGenerator:
    def __init__(self):
        self.fake = fake
        
        # Product categories and brands for realistic data
        self.categories = {
            'Electronics': ['Smartphones', 'Laptops', 'Headphones', 'Tablets', 'Cameras'],
            'Clothing': ['Men Clothing', 'Women Clothing', 'Kids Clothing', 'Shoes', 'Accessories'],
            'Home & Kitchen': ['Furniture', 'Kitchen Appliances', 'Bedding', 'Home Decor', 'Storage'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children Books', 'Comics'],
            'Sports': ['Exercise Equipment', 'Outdoor Gear', 'Sports Apparel', 'Team Sports', 'Water Sports']
        }
        
        self.brands = {
            'Electronics': ['TechPro', 'DigitalMax', 'SmartTech', 'ElectroCore', 'TechWave'],
            'Clothing': ['StyleMax', 'FashionPro', 'TrendWear', 'ComfortFit', 'UrbanStyle'],
            'Home & Kitchen': ['HomePro', 'KitchenMaster', 'ComfortHome', 'ModernLiving', 'CozySpace'],
            'Books': ['BookWorks', 'ReadMore', 'StoryCraft', 'PageTurner', 'BookHouse'],
            'Sports': ['SportsPro', 'FitMax', 'ActiveGear', 'SportElite', 'PowerFit']
        }

    def generate_products(self, n=10000):
        """Generate realistic product data"""
        print(f" Generating {n:,} products...")
        products = []
        
        for i in range(n):
            if i % 2000 == 0:
                print(f"   Generated {i:,} products...")
                
            category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[category])
            brand = random.choice(self.brands[category])
            
            # Price based on category (realistic ranges)
            price_ranges = {
                'Electronics': (25, 1500),
                'Clothing': (15, 250),
                'Home & Kitchen': (12, 600),
                'Books': (8, 45),
                'Sports': (20, 400)
            }
            min_price, max_price = price_ranges[category]
            
            products.append({
                'product_id': str(uuid.uuid4()),
                'asin': f'B{random.randint(10**8, 10**9-1)}',
                'title': f'{brand} {subcategory} {self.fake.word().title()}',
                'category': category,
                'subcategory': subcategory,
                'brand': brand,
                'price': round(random.uniform(min_price, max_price), 2),
                'launch_date': self.fake.date_between(start_date='-3y', end_date='today')
            })
        
        return pd.DataFrame(products)

    def generate_customers(self, n=50000):
        """Generate realistic customer data"""
        print(f" Generating {n:,} customers...")
        customers = []
        countries = ['US', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'India', 'Japan']
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        
        for i in range(n):
            if i % 10000 == 0:
                print(f"   Generated {i:,} customers...")
                
            country = random.choice(countries)
            customers.append({
                'customer_id': str(uuid.uuid4()),
                'customer_code': f'CUST{i+1:07d}',
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'email': self.fake.unique.email(),
                'registration_date': self.fake.date_between(start_date='-3y', end_date='today'),
                'country': country,
                'state': self.fake.state() if country == 'US' else None,
                'city': self.fake.city(),
                'age_group': random.choice(age_groups),
                'gender': random.choice(['Male', 'Female', 'Other'])
            })
        
        return pd.DataFrame(customers)

    def generate_sales(self, products_df, customers_df, n=150000):
        """Generate realistic sales data"""
        print(f" Generating {n:,} sales transactions...")
        sales = []
        payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer']
        order_statuses = ['completed', 'shipped', 'processing', 'cancelled']
        
        # Create some customer preferences (some customers buy more)
        heavy_buyers = random.sample(customers_df['customer_id'].tolist(), int(len(customers_df) * 0.1))
        
        for i in range(n):
            if i % 30000 == 0:
                print(f"   Generated {i:,} sales...")
                
            product = products_df.sample(1).iloc[0]
            
            # Heavy buyers are more likely to be selected
            if random.random() < 0.3 and heavy_buyers:
                customer_id = random.choice(heavy_buyers)
            else:
                customer_id = random.choice(customers_df['customer_id'].tolist())
            
            quantity = max(1, int(np.random.poisson(1.3)))
            
            unit_price = product['price']
            subtotal = unit_price * quantity
            discount_amount = subtotal * random.uniform(0, 0.25)  # 0-25% discount
            shipping_cost = random.uniform(0, 20)
            total_amount = subtotal - discount_amount + shipping_cost
            
            # Status selection
            status_rand = random.random()
            if status_rand < 0.82:
                order_status = 'completed'
            elif status_rand < 0.92:
                order_status = 'shipped'
            elif status_rand < 0.97:
                order_status = 'processing'
            else:
                order_status = 'cancelled'
            
            sales.append({
                'sale_id': str(uuid.uuid4()),
                'product_id': product['product_id'],
                'customer_id': customer_id,
                'order_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': round(total_amount, 2),
                'discount_amount': round(discount_amount, 2),
                'shipping_cost': round(shipping_cost, 2),
                'payment_method': random.choice(payment_methods),
                'shipping_address': self.fake.address().replace('\n', ', '),
                'order_status': order_status
            })
        
        return pd.DataFrame(sales)

    def generate_reviews(self, products_df, customers_df, sales_df, n=200000):
        """Generate realistic review data - FIXED VERSION"""
        print(f" Generating {n:,} reviews...")
        reviews = []
        
        # Get customers who actually made purchases
        purchasing_customers = sales_df['customer_id'].unique()
        
        # Rating distribution (skewed toward positive - realistic for e-commerce)
        ratings = [1, 2, 3, 4, 5]
        
        review_templates = {
            1: ["Terrible product!", "Very disappointed", "Poor quality", "Waste of money", "Don't buy this"],
            2: ["Below expectations", "Not great", "Could be better", "Disappointing", "Not worth it"],
            3: ["Average product", "Okay for the price", "Nothing special", "Decent", "It's fine"],
            4: ["Good product!", "Happy with purchase", "Recommended", "Good value", "Works well"],
            5: ["Excellent!", "Love this product!", "Highly recommend!", "Perfect!", "Amazing quality!"]
        }
        
        for i in range(n):
            if i % 40000 == 0:
                print(f"   Generated {i:,} reviews...")
                
            # Pick a customer who made a purchase (80% of the time)
            if random.random() < 0.8 and len(purchasing_customers) > 0:
                customer_id = random.choice(purchasing_customers)
                
                # Try to pick a product this customer actually bought
                customer_products = sales_df[sales_df['customer_id'] == customer_id]['product_id'].tolist()
                if customer_products and random.random() < 0.7:
                    product_id = random.choice(customer_products)
                else:
                    product_id = random.choice(products_df['product_id'].tolist())
            else:
                customer_id = random.choice(customers_df['customer_id'].tolist())
                product_id = random.choice(products_df['product_id'].tolist())
            
            rating = np.random.choice(ratings, p=[0.04, 0.06, 0.12, 0.38, 0.40])
            review_text = random.choice(review_templates[rating])
            
            # Manual probability for verified purchase
            verified_purchase = random.random() < 0.85
            
            reviews.append({
                'review_id': str(uuid.uuid4()),
                'product_id': product_id,
                'customer_id': customer_id,
                'rating': rating,
                'review_text': review_text,
                'review_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'helpful_votes': max(0, int(np.random.poisson(1.8))),
                'verified_purchase': verified_purchase
            })
        
        return pd.DataFrame(reviews)

    def save_to_csv(self, data_dict, folder='data'):
        """Save all datasets to CSV files"""
        os.makedirs(folder, exist_ok=True)
        
        for name, df in data_dict.items():
            file_path = f'{folder}/{name}.csv'
            df.to_csv(file_path, index=False)
            print(f" Saved {name}: {len(df):,} records to {file_path}")

def main():
    print("Starting E-commerce Data Generation")
    print("=" * 60)
    print("This will create realistic data for the analytics platform:")
    print("• Products with categories, brands, and pricing")
    print("• Customers with demographics and locations") 
    print("• Sales transactions with realistic patterns")
    print("• Product reviews with ratings and text")
    print()
    print("Dataset size: ~410,000 total records")
    print("=" * 60)
    
    generator = EcommerceDataGenerator()
    
    # Generate smaller but still impressive dataset
    print("\n1. Generating Products...")
    products = generator.generate_products(10000)
    
    print("\n2. Generating Customers...")
    customers = generator.generate_customers(50000)
    
    print("\n3. Generating Sales...")
    sales = generator.generate_sales(products, customers, 150000)
    
    print("\n4. Generating Reviews...")
    reviews = generator.generate_reviews(products, customers, sales, 200000)
    
    # Save all data
    print("\n5. Saving Data to CSV Files...")
    data_dict = {
        'products': products,
        'customers': customers, 
        'sales': sales,
        'reviews': reviews
    }
    
    generator.save_to_csv(data_dict)
    
    # Print summary statistics
    total_records = sum(len(df) for df in data_dict.values())
    completed_sales = sales[sales['order_status'] == 'completed']
    total_revenue = completed_sales['total_amount'].sum()
    avg_order_value = completed_sales['total_amount'].mean()
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"  Dataset Summary:")
    print(f"   • Products: {len(products):,}")
    print(f"   • Customers: {len(customers):,}")
    print(f"   • Sales Transactions: {len(sales):,}")
    print(f"   • Product Reviews: {len(reviews):,}")
    print(f"   • Total Records: {total_records:,}")
    print()
    print(f"  Business Metrics:")
    print(f"   • Total Revenue: ${total_revenue:,.2f}")
    print(f"   • Average Order Value: ${avg_order_value:.2f}")
    print(f"   • Completion Rate: {len(completed_sales) / len(sales) * 100:.1f}%")
    print(f"   • Average Rating: {reviews['rating'].mean():.2f}/5.0")
    print()

if __name__ == "__main__":
    main()