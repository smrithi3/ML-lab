#Demonstarte the data cleaning, integration and transformation using pandas library in python
import pandas as pd

customers_data = {
    'customer_id': [1, 2, 3, 4, 5],
    'first_name': ['Tom', 'Sally', 'James', 'Henry', 'James'],
    'last_name': ['Jones', 'Smith', 'Green', 'Black', 'White'],
    'age': [23, 34, 45, 56, 67],
    'city': ['London', None, 'Liverpool', 'Bristol', 'London'],
    'email': ['tom.jones@london.com', 'sally.smith@manchester.com', 'james.green@liverpool.com', 'henry.black@bristol.com', 'james.white@london.com']
}

orders_data = {
    'order_id': [1, 2, 3, 4, 5],
    'customer_id': [1, 2, 3, 4, 5],
    'order_date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
    'amount': [100, 200, 300, 400, 500]
}

products_data = {
    'product_id': [1, 2, 3, 4, 5],
    'product_name': ['Macbook Air M1', 'Samsung Galaxy Notebook', 'Asus ROG Strix 15', 'Dell Inspiron 15', 'LG Gram 16'],
    'price': [75000, 96000, 81000, 56000, 67000]
}

customers_df = pd.DataFrame(customers_data)
orders_df = pd.DataFrame(orders_data)
products_df = pd.DataFrame(products_data)

customers_df['age'] = customers_df['age'].fillna(customers_df['age'].mean())
customers_df['city'] = customers_df['city'].fillna(customers_df['city'].mode()[0])
merged_df = pd.merge(customers_df, orders_df, on='customer_id', how='inner')
merged_df['total_amount'] = merged_df['amount'].sum()

print("Dataset after cleaning, integration and transformation: ")
print(merged_df)