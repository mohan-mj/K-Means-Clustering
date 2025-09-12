# K-Means Clustering: Complete Workshop Project

## Table of Contents
1. [Introduction](#introduction)
2. [Theory and Algorithm](#theory-and-algorithm)
3. [Implementation from Scratch](#implementation-from-scratch)
4. [Project 1: Customer Segmentation](#project-1-customer-segmentation)
5. [Project 2: Image Color Quantization](#project-2-image-color-quantization)
6. [Project 3: Market Segmentation Analysis](#project-3-market-segmentation-analysis)
7. [Advanced Topics](#advanced-topics)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Best Practices and Tips](#best-practices-and-tips)
10. [Exercises and Challenges](#exercises-and-challenges)

---

## Introduction

K-means clustering is one of the most popular unsupervised machine learning algorithms used for partitioning data into k clusters. It's widely applicable in customer segmentation, image processing, market research, and data compression.

### Learning Objectives
- Understand the K-means algorithm and its mathematical foundation
- Implement K-means from scratch
- Apply K-means to real-world datasets
- Learn evaluation techniques and parameter selection
- Understand limitations and best practices

### Prerequisites
- Basic Python programming
- NumPy and pandas familiarity
- Basic understanding of distance metrics
- Matplotlib for visualization

---

## Theory and Algorithm

### How K-Means Works

K-means aims to partition n observations into k clusters where each observation belongs to the cluster with the nearest mean (centroid).

#### Algorithm Steps:
1. **Initialize**: Choose k initial centroids randomly
2. **Assign**: Assign each point to the nearest centroid
3. **Update**: Calculate new centroids as the mean of assigned points
4. **Repeat**: Steps 2-3 until convergence

#### Mathematical Foundation

**Objective Function (Within-Cluster Sum of Squares):**
```
WCSS = Σ(i=1 to k) Σ(x∈Ci) ||x - μi||²
```

Where:
- k = number of clusters
- Ci = set of points in cluster i
- μi = centroid of cluster i
- ||x - μi||² = squared Euclidean distance

**Distance Metric:**
```
d(x, μ) = √(Σ(j=1 to n)(xj - μj)²)
```

### Key Characteristics
- **Partitioning Algorithm**: Creates non-overlapping clusters
- **Centroid-based**: Each cluster represented by its center
- **Distance-based**: Uses Euclidean distance (typically)
- **Iterative**: Converges to local optimum

---

## Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
    def initialize_centroids(self, X):
        """Initialize centroids randomly"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        for i in range(self.k):
            centroid = X[np.random.choice(n_samples)]
            centroids[i] = centroid
            
        return centroids
    
    def calculate_distance(self, X, centroids):
        """Calculate distance between points and centroids"""
        distances = np.zeros((X.shape[0], self.k))
        
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
            
        return distances
    
    def assign_clusters(self, distances):
        """Assign each point to nearest centroid"""
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids based on current assignments"""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                centroids[i] = X[np.random.choice(X.shape[0])]
                
        return centroids
    
    def calculate_wcss(self, X, labels, centroids):
        """Calculate Within-Cluster Sum of Squares"""
        wcss = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss
    
    def fit(self, X):
        """Fit K-means to data"""
        # Initialize
        centroids = self.initialize_centroids(X)
        
        # Store history for visualization
        self.centroids_history = [centroids.copy()]
        self.wcss_history = []
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            distances = self.calculate_distance(X, centroids)
            labels = self.assign_clusters(distances)
            
            # Calculate WCSS
            wcss = self.calculate_wcss(X, labels, centroids)
            self.wcss_history.append(wcss)
            
            # Update centroids
            new_centroids = self.update_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            centroids = new_centroids
            self.centroids_history.append(centroids.copy())
        
        self.centroids = centroids
        self.labels = labels
        self.wcss = wcss
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        distances = self.calculate_distance(X, self.centroids)
        return self.assign_clusters(distances)
    
    def visualize_process(self, X, save_steps=False):
        """Visualize the clustering process"""
        if X.shape[1] != 2:
            print("Visualization only available for 2D data")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        steps = [0, len(self.centroids_history)//3, 2*len(self.centroids_history)//3, -1]
        titles = ['Initialization', 'Early Iteration', 'Mid Iteration', 'Final Result']
        
        for idx, (step, title) in enumerate(zip(steps, titles)):
            ax = axes[idx]
            
            if step == 0:
                # Initial state
                ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
                centroids = self.centroids_history[0]
                ax.scatter(centroids[:, 0], centroids[:, 1], 
                          c='red', marker='x', s=200, linewidths=3)
            else:
                # Clustering state
                centroids = self.centroids_history[step]
                distances = self.calculate_distance(X, centroids)
                labels = self.assign_clusters(distances)
                
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
                for i in range(self.k):
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 0:
                        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                 c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i}')
                
                ax.scatter(centroids[:, 0], centroids[:, 1], 
                          c='black', marker='x', s=200, linewidths=3)
            
            ax.set_title(title)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            if step != 0:
                ax.legend()
        
        plt.tight_layout()
        plt.show()

# Test the implementation
def test_kmeans_implementation():
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                          random_state=42)
    
    # Apply our K-means
    kmeans = KMeans(k=4, random_state=42)
    kmeans.fit(X)
    
    # Visualize the process
    kmeans.visualize_process(X)
    
    # Print results
    print(f"Final WCSS: {kmeans.wcss:.2f}")
    print(f"Number of iterations: {len(kmeans.centroids_history)}")

# Run the test
test_kmeans_implementation()
```

---

## Project 1: Customer Segmentation

### Dataset Creation

```python
# Create synthetic customer data
np.random.seed(42)
n_customers = 1000

# Generate customer features
age = np.random.normal(40, 15, n_customers)
age = np.clip(age, 18, 80)  # Realistic age range

annual_income = np.random.normal(50000, 20000, n_customers)
annual_income = np.clip(annual_income, 20000, 150000)

spending_score = np.random.randint(1, 101, n_customers)

# Create correlations
# Higher income generally means higher spending
income_factor = (annual_income - 20000) / 130000
spending_score = spending_score + np.random.normal(0, 10, n_customers) * income_factor
spending_score = np.clip(spending_score, 1, 100)

# Create DataFrame
customer_data = pd.DataFrame({
    'CustomerID': range(1, n_customers + 1),
    'Age': age.astype(int),
    'Annual_Income': annual_income.astype(int),
    'Spending_Score': spending_score.astype(int)
})

print("Customer Data Sample:")
print(customer_data.head())
print(f"\nDataset Shape: {customer_data.shape}")
print("\nDataset Statistics:")
print(customer_data.describe())
```

### Exploratory Data Analysis

```python
# EDA
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age distribution
axes[0, 0].hist(customer_data['Age'], bins=30, alpha=0.7, color='skyblue')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Income distribution
axes[0, 1].hist(customer_data['Annual_Income'], bins=30, alpha=0.7, color='lightgreen')
axes[0, 1].set_title('Annual Income Distribution')
axes[0, 1].set_xlabel('Annual Income ($)')
axes[0, 1].set_ylabel('Frequency')

# Spending Score distribution
axes[1, 0].hist(customer_data['Spending_Score'], bins=30, alpha=0.7, color='salmon')
axes[1, 0].set_title('Spending Score Distribution')
axes[1, 0].set_xlabel('Spending Score')
axes[1, 0].set_ylabel('Frequency')

# Income vs Spending Score
axes[1, 1].scatter(customer_data['Annual_Income'], customer_data['Spending_Score'], 
                   alpha=0.6, color='purple')
axes[1, 1].set_title('Annual Income vs Spending Score')
axes[1, 1].set_xlabel('Annual Income ($)')
axes[1, 1].set_ylabel('Spending Score')

plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = customer_data[['Age', 'Annual_Income', 'Spending_Score']].corr()
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix')

# Add correlation values
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
```

### Finding Optimal K using Elbow Method

```python
def find_optimal_k(X, k_range=(1, 11)):
    """Find optimal k using elbow method"""
    wcss_values = []
    k_values = range(k_range[0], k_range[1])
    
    for k in k_values:
        kmeans = KMeans(k=k, random_state=42)
        kmeans.fit(X)
        wcss_values.append(kmeans.wcss)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True, alpha=0.3)
    
    # Calculate elbow point using rate of change
    differences = np.diff(wcss_values)
    second_differences = np.diff(differences)
    elbow_index = np.argmax(second_differences) + 2  # +2 because of double diff
    
    plt.axvline(x=k_values[elbow_index], color='red', linestyle='--', 
                label=f'Elbow at k={k_values[elbow_index]}')
    plt.legend()
    plt.show()
    
    return k_values[elbow_index], wcss_values

# Prepare features for clustering
features_for_clustering = customer_data[['Annual_Income', 'Spending_Score']].values

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_clustering)

# Find optimal k
optimal_k, wcss_values = find_optimal_k(features_scaled)
print(f"Suggested optimal k: {optimal_k}")
```

### Customer Segmentation Analysis

```python
# Apply K-means with optimal k
kmeans_customer = KMeans(k=optimal_k, random_state=42)
kmeans_customer.fit(features_scaled)

# Add cluster labels to original data
customer_data['Cluster'] = kmeans_customer.labels

# Analyze clusters
cluster_analysis = customer_data.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual_Income': ['mean', 'std'],
    'Spending_Score': ['mean', 'std'],
    'CustomerID': 'count'
}).round(2)

print("Cluster Analysis:")
print(cluster_analysis)

# Visualize clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(optimal_k):
    cluster_data = customer_data[customer_data['Cluster'] == i]
    plt.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'],
                c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)

# Plot centroids (need to inverse transform)
centroids_original = scaler.inverse_transform(kmeans_customer.centroids)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            c='black', marker='x', s=300, linewidths=3, label='Centroids')

plt.title('Customer Segmentation')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Create customer personas
def create_customer_personas(customer_data):
    personas = {}
    
    for cluster_id in customer_data['Cluster'].unique():
        cluster_data = customer_data[customer_data['Cluster'] == cluster_id]
        
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Annual_Income'].mean()
        avg_spending = cluster_data['Spending_Score'].mean()
        count = len(cluster_data)
        
        # Define persona based on characteristics
        if avg_income < 40000 and avg_spending < 40:
            persona_name = "Budget Conscious"
            description = "Low income, low spending customers who are price-sensitive"
        elif avg_income < 40000 and avg_spending > 60:
            persona_name = "Young Spenders"
            description = "Low income but high spending, likely young professionals"
        elif avg_income > 70000 and avg_spending > 60:
            persona_name = "Premium Customers"
            description = "High income, high spending - our most valuable segment"
        elif avg_income > 70000 and avg_spending < 40:
            persona_name = "Conservative Wealthy"
            description = "High income but conservative spending habits"
        else:
            persona_name = "Middle Class"
            description = "Average income and spending patterns"
        
        personas[cluster_id] = {
            'name': persona_name,
            'description': description,
            'avg_age': round(avg_age, 1),
            'avg_income': round(avg_income),
            'avg_spending': round(avg_spending, 1),
            'count': count,
            'percentage': round(count / len(customer_data) * 100, 1)
        }
    
    return personas

personas = create_customer_personas(customer_data)

print("\n" + "="*60)
print("CUSTOMER PERSONAS")
print("="*60)

for cluster_id, persona in personas.items():
    print(f"\nCluster {cluster_id}: {persona['name']}")
    print(f"Description: {persona['description']}")
    print(f"Average Age: {persona['avg_age']} years")
    print(f"Average Income: ${persona['avg_income']:,}")
    print(f"Average Spending Score: {persona['avg_spending']}")
    print(f"Customer Count: {persona['count']} ({persona['percentage']}%)")
```

---

## Project 2: Image Color Quantization

### Image Color Reduction

```python
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_image(image_path, max_size=(300, 300)):
    """Load and resize image for processing"""
    try:
        # For demo, create a synthetic colorful image if no image provided
        # In practice, you would load: img = Image.open(image_path)
        
        # Create a synthetic image with multiple colors
        img_array = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Add different colored regions
        img_array[0:50, 0:50] = [255, 0, 0]      # Red
        img_array[0:50, 50:100] = [0, 255, 0]    # Green  
        img_array[0:50, 100:150] = [0, 0, 255]   # Blue
        img_array[0:50, 150:200] = [255, 255, 0] # Yellow
        
        img_array[50:100, 0:50] = [255, 0, 255]  # Magenta
        img_array[50:100, 50:100] = [0, 255, 255] # Cyan
        img_array[50:100, 100:150] = [128, 128, 128] # Gray
        img_array[50:100, 150:200] = [255, 128, 0]   # Orange
        
        # Add some noise and gradients
        for i in range(100, 200):
            for j in range(200):
                img_array[i, j] = [
                    int(255 * i / 200),  # Red gradient
                    int(255 * j / 200),  # Green gradient
                    int(128 + 127 * np.sin(i * j * 0.01))  # Blue pattern
                ]
        
        # Add some noise
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return img_array
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def quantize_image_colors(img_array, n_colors=8):
    """Apply K-means clustering to reduce image colors"""
    # Reshape image to be a list of pixels
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, 3)
    
    # Apply K-means
    kmeans = KMeans(k=n_colors, random_state=42)
    kmeans.fit(pixels.astype(float))
    
    # Replace each pixel with its cluster centroid
    quantized_pixels = kmeans.centroids[kmeans.labels]
    quantized_img = quantized_pixels.reshape(h, w, c).astype(np.uint8)
    
    return quantized_img, kmeans.centroids.astype(np.uint8)

def visualize_color_quantization():
    """Demonstrate color quantization with different numbers of colors"""
    # Load image
    original_img = load_and_process_image("sample_image.jpg")
    
    if original_img is None:
        return
    
    # Different numbers of colors to try
    n_colors_list = [2, 4, 8, 16]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Show original
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show quantized versions
    for i, n_colors in enumerate(n_colors_list):
        quantized_img, color_palette = quantize_image_colors(original_img, n_colors)
        
        axes[i+1].imshow(quantized_img)
        axes[i+1].set_title(f'{n_colors} Colors')
        axes[i+1].axis('off')
    
    # Show color palette for the last quantization
    palette_img = np.zeros((50, len(color_palette) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(color_palette):
        palette_img[:, i*50:(i+1)*50] = color
    
    axes[5].imshow(palette_img)
    axes[5].set_title(f'Color Palette ({len(color_palette)} colors)')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original_img, quantized_img, color_palette

# Run color quantization demo
original_img, quantized_img, color_palette = visualize_color_quantization()

# Analyze compression
original_unique_colors = len(np.unique(original_img.reshape(-1, 3), axis=0))
quantized_unique_colors = len(color_palette)

print(f"Original image had {original_unique_colors} unique colors")
print(f"Quantized image has {quantized_unique_colors} unique colors")
print(f"Compression ratio: {original_unique_colors / quantized_unique_colors:.1f}:1")
```

---

## Project 3: Market Segmentation Analysis

### Synthetic Market Data

```python
def create_market_research_data(n_respondents=500):
    """Create synthetic market research survey data"""
    np.random.seed(42)
    
    # Define customer segments in advance (for validation)
    segments = {
        'Tech Enthusiasts': 0.25,
        'Budget Conscious': 0.30,
        'Luxury Seekers': 0.20,
        'Eco-Friendly': 0.25
    }
    
    data = []
    segment_labels = []
    
    for segment, proportion in segments.items():
        n_segment = int(n_respondents * proportion)
        
        for _ in range(n_segment):
            if segment == 'Tech Enthusiasts':
                record = {
                    'age': np.random.normal(28, 5),
                    'income': np.random.normal(75000, 15000),
                    'tech_interest': np.random.normal(9, 1),
                    'price_sensitivity': np.random.normal(4, 2),
                    'brand_loyalty': np.random.normal(6, 2),
                    'eco_consciousness': np.random.normal(6, 2),
                    'social_influence': np.random.normal(7, 2)
                }
            elif segment == 'Budget Conscious':
                record = {
                    'age': np.random.normal(45, 10),
                    'income': np.random.normal(45000, 12000),
                    'tech_interest': np.random.normal(5, 2),
                    'price_sensitivity': np.random.normal(9, 1),
                    'brand_loyalty': np.random.normal(4, 2),
                    'eco_consciousness': np.random.normal(5, 2),
                    'social_influence': np.random.normal(4, 2)
                }
            elif segment == 'Luxury Seekers':
                record = {
                    'age': np.random.normal(38, 8),
                    'income': np.random.normal(95000, 20000),
                    'tech_interest': np.random.normal(6, 2),
                    'price_sensitivity': np.random.normal(3, 1),
                    'brand_loyalty': np.random.normal(9, 1),
                    'eco_consciousness': np.random.normal(4, 2),
                    'social_influence': np.random.normal(8, 2)
                }
            else:  # Eco-Friendly
                record = {
                    'age': np.random.normal(35, 7),
                    'income': np.random.normal(65000, 18000),
                    'tech_interest': np.random.normal(6, 2),
                    'price_sensitivity': np.random.normal(6, 2),
                    'brand_loyalty': np.random.normal(7, 2),
                    'eco_consciousness': np.random.normal(9, 1),
                    'social_influence': np.random.normal(6, 2)
                }
            
            # Ensure values are in reasonable ranges
            for key in record:
                if key == 'age':
                    record[key] = np.clip(record[key], 18, 70)
                elif key == 'income':
                    record[key] = np.clip(record[key], 25000, 150000)
                else:  # Likert scale items
                    record[key] = np.clip(record[key], 1, 10)
            
            data.append(record)
            segment_labels.append(segment)
    
    # Convert to DataFrame
    market_df = pd.DataFrame(data)
    market_df['true_segment'] = segment_labels
    
    return market_df

# Create market research data
market_data = create_market_research_data()

print("Market Research Data Sample:")
print(market_data.head())
print(f"\nDataset Shape: {market_data.shape}")
print("\nTrue Segments Distribution:")
print(market_data['true_segment'].value_counts())
```

### Market Segmentation with K-means

```python
# Prepare features for clustering (exclude true labels)
feature_columns = ['age', 'income', 'tech_interest', 'price_sensitivity', 
                  'brand_loyalty', 'eco_consciousness', 'social_influence']

X_market = market_data[feature_columns].values

# Standardize features
scaler_market = StandardScaler()
X_market_scaled = scaler_market.fit_transform(X_market)

# Find optimal number of clusters
optimal_k_market, wcss_market = find_optimal_k(X_market_scaled, k_range=(2, 8))

# Apply K-means
kmeans_market = KMeans(k=4, random_state=42)  # We know there are 4 true segments
kmeans_market.fit(X_market_scaled)

# Add predictions to dataframe
market_data['predicted_cluster'] = kmeans_market.labels

# Analyze discovered segments
segment_profiles = market_data.groupby('predicted_cluster')[feature_columns].mean()
segment_sizes = market_data.groupby('predicted_cluster').size()

print("Discovered Market Segments:")
print("="*50)
print(segment_profiles.round(2))
print(f"\nSegment Sizes: {segment_sizes.values}")

# Create radar chart for segment profiles
def create_radar_chart(segment_profiles):
    """Create radar chart to visualize segment profiles"""
    categories = feature_columns
    n_segments = len(segment_profiles)
    
    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = ['red', 'blue', 'green', 'purple']
    
    for i in range(n_segments):
        values = segment_profiles.iloc[i].values.tolist()
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Segment {i}', color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 10)
    ax.set_title('Market Segment Profiles', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

create_radar_chart(segment_profiles)

# Compare with true segments
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Calculate clustering performance metrics
ari_score = adjusted_rand_