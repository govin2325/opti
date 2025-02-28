import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create sample customer data (or load your own data)
def generate_sample_data(n_customers=1000):
    """Generate synthetic customer purchase data"""
    data = {
        'customer_id': range(1, n_customers + 1),
        'recency': np.random.randint(1, 365, size=n_customers),  # days since last purchase
        'frequency': np.random.exponential(scale=5, size=n_customers).astype(int) + 1,  # number of purchases
        'monetary': np.random.exponential(scale=500, size=n_customers),  # average purchase value
        'tenure': np.random.randint(1, 1000, size=n_customers),  # days as customer
        'age': np.random.normal(loc=40, scale=12, size=n_customers).astype(int),
        'product_categories': np.random.randint(1, 10, size=n_customers)  # number of different categories purchased
    }
    
    # Add correlations to make data more realistic
    data['monetary'] = data['monetary'] * (0.5 + 0.5 * data['frequency'] / data['frequency'].max())
    
    return pd.DataFrame(data)

# Load or generate data
try:
    # Try to load your own data
    # df = pd.read_csv('customer_data.csv')
    # If no file, generate sample data
    df = generate_sample_data(1000)
    print("Sample data generated with 1000 customers")
except:
    df = generate_sample_data(1000)
    print("Sample data generated with 1000 customers")

# 2. Data exploration and preprocessing
print("\nData Overview:")
print(df.head())
print("\nData Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# RFM Analysis (Recency, Frequency, Monetary Value)
rfm_data = df[['recency', 'frequency', 'monetary']]

# Handle outliers using capping (optional)
for col in rfm_data.columns:
    q1 = rfm_data[col].quantile(0.05)
    q3 = rfm_data[col].quantile(0.95)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    rfm_data[col] = np.where(rfm_data[col] < lower_bound, lower_bound, 
                     np.where(rfm_data[col] > upper_bound, upper_bound, rfm_data[col]))

# 3. Feature scaling (standardization)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_data.columns)

# 4. Determine optimal number of clusters using Elbow Method
wcss = []  # Within-Cluster Sum of Squares
silhouette_scores = []
max_clusters = 10

for i in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Score Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.savefig('optimal_clusters.png')
plt.show()

# Determine the optimal number of clusters (usually where the elbow occurs)
# For this example, let's say we found the optimal k to be 5
optimal_k = 5  # This should be determined from the above plots
print(f"\nOptimal number of clusters based on elbow method: {optimal_k}")

# 5. Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(rfm_scaled)

# 6. Analyze the clusters
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                              columns=rfm_data.columns)
cluster_centers.index.name = 'Cluster'
print("\nCluster Centers (Original Scale):")
print(cluster_centers)

# Calculate cluster statistics
cluster_stats = df.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'tenure': 'mean',
    'age': 'mean',
    'product_categories': 'mean',
    'customer_id': 'count'
}).rename(columns={'customer_id': 'count'})

print("\nCluster Statistics:")
print(cluster_stats)

# 7. Visualize the clusters using PCA to reduce to 2D
pca = PCA(n_components=2)
principal_components = pca.fit_transform(rfm_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['cluster']

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=60)
plt.title('Customer Segments Visualization (PCA)')
plt.legend(title='Cluster')
plt.savefig('customer_segments_pca.png')
plt.show()

# 8. Create a 3D plot for better visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
scatter = ax.scatter(rfm_scaled_df['recency'], 
                     rfm_scaled_df['frequency'], 
                     rfm_scaled_df['monetary'],
                     c=df['cluster'], 
                     cmap='viridis', 
                     s=50, 
                     alpha=0.7)

# Add legend and labels
ax.set_xlabel('Recency (Standardized)')
ax.set_ylabel('Frequency (Standardized)')
ax.set_zlabel('Monetary Value (Standardized)')
ax.set_title('3D Visualization of Customer Segments')

legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
plt.savefig('customer_segments_3d.png')
plt.show()

# 9. Perform more detailed analysis on each cluster
for i in range(optimal_k):
    cluster_customers = df[df['cluster'] == i]
    print(f"\nCluster {i} Analysis:")
    print(f"Number of customers: {len(cluster_customers)}")
    print(f"Percentage of total: {len(cluster_customers) / len(df) * 100:.2f}%")
    print("\nKey metrics (mean values):")
    print(f"Recency: {cluster_customers['recency'].mean():.2f} days")
    print(f"Frequency: {cluster_customers['frequency'].mean():.2f} purchases")
    print(f"Monetary: ${cluster_customers['monetary'].mean():.2f}")
    print(f"Tenure: {cluster_customers['tenure'].mean():.2f} days")
    print(f"Age: {cluster_customers['age'].mean():.2f} years")
    print(f"Product Categories: {cluster_customers['product_categories'].mean():.2f}")

# 10. Create radar charts to visualize cluster characteristics
from math import pi

# Create a function for radar charts
def create_radar_chart(cluster_data, cluster_names):
    # Number of variables
    categories = list(cluster_data.columns)
    N = len(categories)
    
    # Create angle for each variable
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    
    # Add each cluster
    for i, cluster_name in enumerate(cluster_names):
        values = cluster_data.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Customer Segments Characteristics', size=16)
    plt.savefig('customer_segments_radar.png')
    plt.show()

# Prepare data for radar chart (normalize the cluster centers)
radar_data = cluster_stats.drop('count', axis=1)
# Min-max scaling for radar chart
for col in radar_data.columns:
    radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())

# Create radar chart
create_radar_chart(radar_data, [f'Cluster {i}' for i in range(optimal_k)])

# 11. Assign meaningful labels to clusters based on their characteristics
def assign_segment_labels(cluster_stats):
    segment_labels = {}
    
    for cluster_id in cluster_stats.index:
        row = cluster_stats.loc[cluster_id]
        
        # High monetary value, high frequency, low recency
        if row['monetary'] > cluster_stats['monetary'].mean() and \
           row['frequency'] > cluster_stats['frequency'].mean() and \
           row['recency'] < cluster_stats['recency'].mean():
            segment_labels[cluster_id] = "High-Value Loyal Customers"
        
        # High monetary, low frequency, low recency
        elif row['monetary'] > cluster_stats['monetary'].mean() and \
             row['frequency'] < cluster_stats['frequency'].mean() and \
             row['recency'] < cluster_stats['recency'].mean():
            segment_labels[cluster_id] = "Big Spenders (Infrequent)"
        
        # Low monetary, high frequency, low recency
        elif row['monetary'] < cluster_stats['monetary'].mean() and \
             row['frequency'] > cluster_stats['frequency'].mean() and \
             row['recency'] < cluster_stats['recency'].mean():
            segment_labels[cluster_id] = "Frequent Low-Value Shoppers"
        
        # High recency (haven't purchased recently)
        elif row['recency'] > cluster_stats['recency'].mean() * 1.5:
            segment_labels[cluster_id] = "At-Risk Customers"
        
        # New customers (low tenure)
        elif row['tenure'] < cluster_stats['tenure'].mean() * 0.5:
            segment_labels[cluster_id] = "New Customers"
        
        # Default label
        else:
            segment_labels[cluster_id] = f"Segment {cluster_id}"
    
    return segment_labels

# Assign segments
segment_labels = assign_segment_labels(cluster_stats)
print("\nCluster Segments:")
for cluster_id, label in segment_labels.items():
    print(f"Cluster {cluster_id}: {label}")

# Add segment labels to the dataframe
df['segment'] = df['cluster'].map(segment_labels)

# 12. Generate marketing recommendations for each segment
print("\nMarketing Recommendations:")
for cluster_id, label in segment_labels.items():
    print(f"\nFor {label} (Cluster {cluster_id}):")
    
    if "High-Value Loyal" in label:
        print("- Implement a premium loyalty program with exclusive benefits")
        print("- Offer early access to new products and services")
        print("- Provide personal shopping/concierge services")
    
    elif "Big Spenders" in label:
        print("- Create bundles or volume discounts to encourage more frequent purchases")
        print("- Implement a 'coming back' reward for purchases within a shorter time frame")
        print("- Send personalized product recommendations based on past high-value purchases")
    
    elif "Frequent Low-Value" in label:
        print("- Introduce upselling strategies to increase average order value")
        print("- Create tiered rewards that incentivize higher spending")
        print("- Offer limited-time promotions for premium products/services")
    
    elif "At-Risk" in label:
        print("- Launch a win-back campaign with special offers")
        print("- Request feedback through surveys to understand reasons for absence")
        print("- Create a re-engagement email sequence with escalating incentives")
    
    elif "New" in label:
        print("- Create an onboarding journey to familiarize customers with your offerings")
        print("- Offer first-time purchase incentives for a second purchase")
        print("- Provide educational content about products/services")
    
    else:
        print("- Analyze specific characteristics to create targeted marketing strategies")
        print("- Test different promotional approaches to determine optimal engagement")

# 13. Save the results
df.to_csv('customer_segments_results.csv', index=False)
print("\nResults saved to 'customer_segments_results.csv'")

# 14. Final distribution of customers across segments
plt.figure(figsize=(12, 6))
segment_counts = df['segment'].value_counts()
segment_counts.plot(kind='bar', color=sns.color_palette('viridis', len(segment_counts)))
plt.title('Distribution of Customers Across Segments')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('segment_distribution.png')
plt.show()

print("\nCustomer segmentation analysis complete!")