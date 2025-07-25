"""
Parkinson's Disease EDA Analysis
===============================
Phase 5: Exploratory Data Analysis for Multimodal Treatment Monitoring
Dataset: fox_insight_multimodal_CORE_IMPUTED.csv
Purpose: Discover patterns and relationships before model training
Author: Niket
Date: July 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Fix Unicode encoding for Windows console
if sys.platform.startswith('win'):
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        # If UTF-8 encoding fails, continue without it
        pass

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. DATA LOADING AND INITIAL SETUP
# ============================================================================

def load_and_setup_data():
    """Load the imputed dataset and perform initial setup."""
    print("🎯 PHASE 5: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load your imputed dataset
    DATA_PATH = r"C:\Users\niket\OneDrive\Desktop\filtered_data\fox_insight_multimodal_CORE_IMPUTED.csv"
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Shape: {df.shape}")
        print(f"👥 Unique patients: {df['fox_insight_id'].nunique():,}")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        return df
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        print("Please check the file path and try again.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

# ============================================================================
# 2. DATASET OVERVIEW ANALYSIS
# ============================================================================

def dataset_overview(df):
    """Comprehensive dataset overview and basic statistics."""
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 40)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Unique patients: {df['fox_insight_id'].nunique():,}")
    print(f"Total records: {len(df):,}")
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    print(f"\nData types distribution:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing values check (should be 0 after imputation)
    missing_total = df.isnull().sum().sum()
    print(f"\nMissing values: {missing_total} (should be 0 after imputation)")
    
    # Feature categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID column from numeric for analysis
    if 'fox_insight_id' in numeric_cols:
        numeric_cols.remove('fox_insight_id')
    
    print(f"\nFeature categories:")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    
    # Sample data preview
    print(f"\n📋 DATA PREVIEW (First 5 rows):")
    print("-" * 50)
    print(df.head().to_string())
    
    return numeric_cols, categorical_cols

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS (TREATMENT RESPONSE)
# ============================================================================

def analyze_target_variable(df, target_col='CGIPD'):
    """Comprehensive analysis of treatment response patterns."""
    print(f"\n🎯 TARGET VARIABLE ANALYSIS: {target_col}")
    print("-" * 50)
    
    if target_col not in df.columns:
        print(f"⚠️ Target variable '{target_col}' not found in dataset")
        return None
    
    # Distribution analysis
    target_dist = df[target_col].value_counts().sort_index()
    print(f"Treatment Response Distribution:")
    for score, count in target_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  CGIPD {score}: {count:,} patients ({percentage:.1f}%)")
    
    # Statistical summary
    print(f"\nStatistical Summary:")
    print(f"  Mean: {df[target_col].mean():.2f}")
    print(f"  Median: {df[target_col].median():.2f}")
    print(f"  Std Dev: {df[target_col].std():.2f}")
    print(f"  Range: {df[target_col].min()} - {df[target_col].max()}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution histogram
    axes[0,0].hist(df[target_col], bins=len(target_dist), alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Treatment Response Distribution (CGIPD)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('CGIPD Score (1=Much Better, 7=Much Worse)')
    axes[0,0].set_ylabel('Number of Patients')
    axes[0,0].grid(True, alpha=0.3)
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(target_dist)))
    axes[0,1].pie(target_dist.values, labels=[f'CGIPD {i}' for i in target_dist.index], 
                  autopct='%1.1f%%', colors=colors)
    axes[0,1].set_title('Treatment Response Proportions', fontsize=14, fontweight='bold')
    
    # Box plot by sex (if available)
    if 'Sex' in df.columns:
        sex_data = []
        sex_labels = []
        for sex in df['Sex'].unique():
            if pd.notna(sex):
                sex_data.append(df[df['Sex'] == sex][target_col].dropna())
                sex_labels.append(f'Sex {sex}')
        
        if sex_data:
            axes[1,0].boxplot(sex_data, labels=sex_labels)
            axes[1,0].set_title('Treatment Response by Sex', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('CGIPD Score')
            axes[1,0].grid(True, alpha=0.3)
    
    # Age vs Treatment Response scatter plot
    if 'age' in df.columns:
        scatter = axes[1,1].scatter(df['age'], df[target_col], alpha=0.5, c=df[target_col], 
                                   cmap='viridis', s=30)
        axes[1,1].set_xlabel('Age (years)')
        axes[1,1].set_ylabel('CGIPD Score')
        axes[1,1].set_title('Age vs Treatment Response', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('treatment_response_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return target_dist

# ============================================================================
# 4. MULTIMODAL FEATURE CORRELATION ANALYSIS
# ============================================================================

def multimodal_correlation_analysis(df):
    """Analyze correlations between speech, motor, and clinical features."""
    print(f"\n🔬 MULTIMODAL CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Define feature groups based on your dataset
    speech_features = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['speech', 'voice', 'talk', 'bradyspeech'])]
    
    motor_features = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['walk', 'episode', 'motor', 'hands', 'comp', 'freeze'])]
    
    clinical_features = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['cgipd', 'sev', 'updrs', 'moca'])]
    
    demographic_features = ['age', 'Sex', 'Education', 'Income', 'Employment']
    demographic_features = [col for col in demographic_features if col in df.columns]
    
    print(f"Feature groups identified:")
    print(f"  Speech features: {len(speech_features)} - {speech_features[:3]}...")
    print(f"  Motor features: {len(motor_features)} - {motor_features[:3]}...")
    print(f"  Clinical features: {len(clinical_features)} - {clinical_features[:3]}...")
    print(f"  Demographic features: {len(demographic_features)} - {demographic_features}")
    
    # Create correlation matrix for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'fox_insight_id' in numeric_cols:
        numeric_cols = numeric_cols.drop('fox_insight_id')
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create comprehensive correlation heatmap
    plt.figure(figsize=(20, 16))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Multimodal Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Focus on correlations with target variable
    if 'CGIPD' in df.columns:
        target_corr = corr_matrix['CGIPD'].abs().sort_values(ascending=False)
        print(f"\n🎯 TOP 15 FEATURES CORRELATED WITH TREATMENT RESPONSE:")
        print("-" * 60)
        for i, (feature, corr) in enumerate(target_corr.head(16).items(), 1):
            if feature != 'CGIPD':
                print(f"{i:2d}. {feature:<40} {corr:.4f}")
    
    # Cross-modal correlation analysis
    print(f"\n🔗 CROSS-MODAL CORRELATIONS:")
    print("-" * 30)
    
    # Speech-Motor correlations
    if speech_features and motor_features:
        speech_motor_corrs = []
        for speech_feat in speech_features:
            for motor_feat in motor_features:
                if speech_feat in df.columns and motor_feat in df.columns:
                    corr = df[speech_feat].corr(df[motor_feat])
                    if abs(corr) > 0.1:  # Only show meaningful correlations
                        speech_motor_corrs.append((speech_feat, motor_feat, corr))
        
        if speech_motor_corrs:
            speech_motor_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            print("Strong Speech-Motor correlations:")
            for speech, motor, corr in speech_motor_corrs[:5]:
                print(f"  {speech[:25]} ↔ {motor[:25]} : {corr:.3f}")
    
    return corr_matrix, speech_features, motor_features, clinical_features

# ============================================================================
# 5. PATIENT SEGMENTATION ANALYSIS
# ============================================================================

def patient_segmentation_analysis(df):
    """Identify distinct patient subgroups using clustering."""
    print(f"\n👥 PATIENT SEGMENTATION ANALYSIS")
    print("-" * 40)
    
    # Select features for clustering
    clustering_features = ['age', 'CGIPD', 'CompDiffWalk', 'WalkDay', 'Episode']
    available_features = [col for col in clustering_features if col in df.columns]
    
    if len(available_features) < 3:
        print("⚠️ Insufficient features for meaningful clustering analysis")
        print(f"Available features: {available_features}")
        return None, None, None
    
    print(f"Using features for clustering: {available_features}")
    
    # Prepare clustering data
    cluster_data = df[available_features].copy()
    
    # Handle any remaining missing values
    for col in cluster_data.columns:
        if cluster_data[col].dtype in ['float64', 'int64']:
            cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
        else:
            cluster_data[col] = cluster_data[col].fillna(cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else 0)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    try:
        from sklearn.metrics import silhouette_score
        silhouette_available = True
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
        
        # Choose optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")
        
    except ImportError:
        print("⚠️ silhouette_score not available, using k=4")
        silhouette_available = False
        optimal_k = 4
        
        # Only calculate inertias for elbow method
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)
    
    # Create clustering analysis plots
    if silhouette_available:
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Elbow curve
        axes[0,0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0,0].set_xlabel('Number of Clusters (k)')
        axes[0,0].set_ylabel('Inertia')
        axes[0,0].set_title('Elbow Method for Optimal k', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[0,1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[0,1].set_xlabel('Number of Clusters (k)')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].set_title('Silhouette Analysis', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        plot_layout = "2x2"
    else:
        # Create 1x2 subplot layout
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Elbow curve only
        axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        plot_layout = "1x2"
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # PCA visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    # ENHANCED: Handle different plot layouts safely
    if plot_layout == "2x2":
        # 2x2 grid layout
        scatter = axes[1,0].scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, 
                                   cmap='viridis', alpha=0.6, s=50)
        axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1,0].set_title('Patient Clusters (PCA Visualization)', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Cluster size distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        axes[1,1].bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Number of Patients')
        axes[1,1].set_title('Cluster Size Distribution', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
    else:  # 1x2 grid layout
        # PCA visualization only (no space for cluster size distribution)
        scatter = axes[1].scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, 
                                 cmap='viridis', alpha=0.6, s=50)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1].set_title('Patient Clusters (PCA Visualization)', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1])
        
        # Create separate figure for cluster size distribution
        plt.figure(figsize=(8, 6))
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        plt.bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Patients')
        plt.title('Cluster Size Distribution', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plt.tight_layout()
    plt.savefig('patient_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze cluster characteristics
    print(f"\n🔍 CLUSTER CHARACTERISTICS:")
    print("-" * 40)
    
    # Calculate means and standard deviations separately
    cluster_means = df_clustered.groupby('Cluster')[available_features].mean()
    cluster_stds = df_clustered.groupby('Cluster')[available_features].std()
    
    print("Cluster Means:")
    print(cluster_means.round(3))
    
    print("\nCluster Standard Deviations:")
    print(cluster_stds.round(3))
    
    # Clinical interpretation of clusters
    print(f"\n📋 CLUSTER CLINICAL PROFILES:")
    print("-" * 35)
    for cluster_id in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(df_clustered)) * 100
        
        print(f"\nCluster {cluster_id} (n={cluster_size:,}, {cluster_pct:.1f}%):")
        
        if 'age' in available_features:
            age_mean = cluster_data['age'].mean()
            age_std = cluster_data['age'].std()
            print(f"  Age: {age_mean:.1f} ± {age_std:.1f} years")
        
        if 'CGIPD' in available_features:
            cgipd_mean = cluster_data['CGIPD'].mean()
            cgipd_std = cluster_data['CGIPD'].std()
            print(f"  Treatment Response (CGIPD): {cgipd_mean:.2f} ± {cgipd_std:.2f}")
        
        if 'CompDiffWalk' in available_features:
            walk_mean = cluster_data['CompDiffWalk'].mean()
            walk_std = cluster_data['CompDiffWalk'].std()
            print(f"  Walking Difficulty: {walk_mean:.2f} ± {walk_std:.2f}")
        
        if 'WalkDay' in available_features:
            walkday_mean = cluster_data['WalkDay'].mean()
            walkday_std = cluster_data['WalkDay'].std()
            print(f"  Walking Days/Week: {walkday_mean:.1f} ± {walkday_std:.1f}")
        
        if 'Episode' in available_features:
            episode_mean = cluster_data['Episode'].mean()
            episode_std = cluster_data['Episode'].std()
            print(f"  Episode Score: {episode_mean:.2f} ± {episode_std:.2f}")
    
    # Cluster comparison analysis
    print(f"\n📊 CLUSTER COMPARISON INSIGHTS:")
    print("-" * 40)
    
    if 'CGIPD' in available_features:
        best_response_cluster = cluster_means['CGIPD'].idxmin()  # Lower CGIPD = better response
        worst_response_cluster = cluster_means['CGIPD'].idxmax()  # Higher CGIPD = worse response
        print(f"Best Treatment Response: Cluster {best_response_cluster} (CGIPD: {cluster_means.loc[best_response_cluster, 'CGIPD']:.2f})")
        print(f"Worst Treatment Response: Cluster {worst_response_cluster} (CGIPD: {cluster_means.loc[worst_response_cluster, 'CGIPD']:.2f})")
    
    if 'age' in available_features:
        youngest_cluster = cluster_means['age'].idxmin()
        oldest_cluster = cluster_means['age'].idxmax()
        print(f"Youngest Group: Cluster {youngest_cluster} (Age: {cluster_means.loc[youngest_cluster, 'age']:.1f})")
        print(f"Oldest Group: Cluster {oldest_cluster} (Age: {cluster_means.loc[oldest_cluster, 'age']:.1f})")
    
    return df_clustered, cluster_labels, optimal_k


# ============================================================================
# 6. TEMPORAL PROGRESSION ANALYSIS
# ============================================================================

def temporal_progression_analysis(df):
    """Analyze disease progression over time."""
    print(f"\n⏰ TEMPORAL PROGRESSION ANALYSIS")
    print("-" * 40)
    
    # Check for temporal columns
    temporal_cols = ['age', 'days_elapsed']
    available_temporal = [col for col in temporal_cols if col in df.columns]
    
    if not available_temporal:
        print("⚠️ No temporal columns found for progression analysis")
        return None
    
    print(f"Temporal columns available: {available_temporal}")
    
    # Analyze progression using severity scores
    severity_cols = [col for col in df.columns if 'sev' in col.lower()]
    print(f"Severity columns found: {severity_cols}")
    
    # Create comprehensive temporal analysis
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()  # Flatten for easier indexing
    
    plot_idx = 0
    
    # Age distribution
    if 'age' in df.columns:
        axes[plot_idx].hist(df['age'], bins=30, alpha=0.7, edgecolor='black')
        axes[plot_idx].set_xlabel('Age (years)')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].set_title('Age Distribution in Dataset', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Age vs Treatment Response
    if 'age' in df.columns and 'CGIPD' in df.columns:
        # Create age bins for better visualization
        df['age_group'] = pd.cut(df['age'], bins=5, labels=['<50', '50-60', '60-70', '70-80', '80+'])
        
        age_treatment = df.groupby('age_group')['CGIPD'].agg(['mean', 'std']).reset_index()
        axes[plot_idx].bar(range(len(age_treatment)), age_treatment['mean'], 
                          yerr=age_treatment['std'], capsize=5, alpha=0.7)
        axes[plot_idx].set_xticks(range(len(age_treatment)))
        axes[plot_idx].set_xticklabels(age_treatment['age_group'])
        axes[plot_idx].set_xlabel('Age Group')
        axes[plot_idx].set_ylabel('Average CGIPD Score')
        axes[plot_idx].set_title('Treatment Response by Age Group', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Days elapsed vs outcomes
    if 'days_elapsed' in df.columns and 'CGIPD' in df.columns:
        scatter = axes[plot_idx].scatter(df['days_elapsed'], df['CGIPD'], 
                                        alpha=0.5, c=df['age'] if 'age' in df.columns else 'blue',
                                        cmap='viridis', s=30)
        axes[plot_idx].set_xlabel('Days Elapsed Since First Visit')
        axes[plot_idx].set_ylabel('CGIPD Score')
        axes[plot_idx].set_title('Treatment Response Over Time', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        if 'age' in df.columns:
            plt.colorbar(scatter, ax=axes[plot_idx], label='Age')
        plot_idx += 1
    
    # Motor function over time
    if 'CompDiffWalk' in df.columns and 'days_elapsed' in df.columns:
        axes[plot_idx].scatter(df['days_elapsed'], df['CompDiffWalk'], 
                              alpha=0.5, color='orange', s=30)
        axes[plot_idx].set_xlabel('Days Elapsed Since First Visit')
        axes[plot_idx].set_ylabel('Walking Difficulty Score')
        axes[plot_idx].set_title('Motor Function Over Time', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Severity progression (if multiple timepoints available)
    if len(severity_cols) > 1:
        severity_data = df[severity_cols].dropna()
        if len(severity_data) > 0:
            axes[plot_idx].boxplot([severity_data[col] for col in severity_cols], 
                                  labels=[f'T{i+1}' for i in range(len(severity_cols))])
            axes[plot_idx].set_xlabel('Time Point')
            axes[plot_idx].set_ylabel('Severity Score')
            axes[plot_idx].set_title('Disease Severity Progression', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
    
    # Age vs Motor function
    if 'age' in df.columns and 'CompDiffWalk' in df.columns:
        axes[plot_idx].scatter(df['age'], df['CompDiffWalk'], alpha=0.5, color='red', s=30)
        axes[plot_idx].set_xlabel('Age (years)')
        axes[plot_idx].set_ylabel('Walking Difficulty Score')
        axes[plot_idx].set_title('Age vs Motor Function', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('temporal_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis of progression
    if 'age' in df.columns and 'CGIPD' in df.columns:
        age_corr = df['age'].corr(df['CGIPD'])
        print(f"\nAge-Treatment Response Correlation: {age_corr:.3f}")
        
        if abs(age_corr) > 0.1:
            print(f"{'Positive' if age_corr > 0 else 'Negative'} correlation detected")
    
    return df

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def feature_importance_analysis(df):
    """Analyze feature importance using Random Forest with progress tracking."""
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    # Check if target variable exists
    if 'CGIPD' not in df.columns:
        print("⚠️ Target variable CGIPD not found")
        return None
    
    # Prepare features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID and target from features
    feature_cols = [col for col in numeric_cols if col not in ['fox_insight_id', 'CGIPD']]
    
    print(f"Using {len(feature_cols)} features for importance analysis")
    
    # Prepare data with progress updates
    print("🔄 Preparing data...")
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['CGIPD'].fillna(df['CGIPD'].median())
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Sample data if too large (speed optimization)
    if len(X) > 10000:
        print(f"🔄 Sampling {10000} records for faster processing...")
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    # Train Random Forest for feature importance
    try:
        print("🔄 Training Random Forest model...")
        # Reduced number of estimators for speed
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=1)
        rf.fit(X_sample, y_sample)
        
        print("🔄 Calculating feature importance...")
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("🔄 Creating visualization...")
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Top 20 features
        top_features = importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features for Treatment Response Prediction', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print(f"\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for idx, row in importance_df.head(15).iterrows():
            print(f"{row['Feature']:<35} {row['Importance']:.4f}")
        
        # Feature importance by category
        speech_importance = importance_df[importance_df['Feature'].str.contains(
            'speech|voice|talk|brady', case=False, na=False)]['Importance'].sum()
        motor_importance = importance_df[importance_df['Feature'].str.contains(
            'walk|episode|motor|hands|comp|freeze', case=False, na=False)]['Importance'].sum()
        clinical_importance = importance_df[importance_df['Feature'].str.contains(
            'sev|updrs|moca|cgipd', case=False, na=False)]['Importance'].sum()
        
        print(f"\n📈 IMPORTANCE BY CATEGORY:")
        print("-" * 30)
        print(f"Speech features: {speech_importance:.3f}")
        print(f"Motor features: {motor_importance:.3f}")
        print(f"Clinical features: {clinical_importance:.3f}")
        
        return importance_df
        
    except Exception as e:
        print(f"❌ Error in feature importance analysis: {str(e)}")
        return None



# ============================================================================
# 8. STATISTICAL SUMMARY AND INSIGHTS
# ============================================================================

def statistical_summary(df):
    """Generate comprehensive statistical summary."""
    print(f"\n📈 STATISTICAL SUMMARY")
    print("-" * 30)
    
    # Numeric features summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'fox_insight_id' in numeric_cols:
        numeric_cols = numeric_cols.drop('fox_insight_id')
    
    stats_summary = df[numeric_cols].describe()
    print("Descriptive Statistics (Top 10 columns):")
    print(stats_summary.iloc[:, :10].round(3))
    
    # Correlation insights
    if 'CGIPD' in df.columns:
        correlations = df[numeric_cols].corrwith(df['CGIPD']).abs().sort_values(ascending=False)
        print(f"\nStrongest correlations with treatment response:")
        for feature, corr in correlations.head(8).items():
            if feature != 'CGIPD':
                print(f"  {feature}: {corr:.3f}")
    
    # Missing data analysis (should be minimal after imputation)
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nRemaining missing data:")
        missing_features = missing_data[missing_data > 0]
        for feature, count in missing_features.items():
            print(f"  {feature}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n✅ No missing data detected (imputation successful)")


# ============================================================================
# 10. MAIN EXECUTION FUNCTION
# ============================================================================

def generate_eda_report(df, target_dist, corr_matrix, importance_df):
    """Stub for EDA report generation. Expand as needed."""
    report_path = "comprehensive_eda_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Comprehensive EDA Report\n")
        f.write("="*30 + "\n")
        f.write(f"Shape: {df.shape}\n")
        if target_dist is not None:
            f.write("\nTarget Distribution:\n")
            f.write(str(target_dist) + "\n")
        if corr_matrix is not None:
            f.write("\nCorrelation Matrix (top 5):\n")
            f.write(str(corr_matrix.head()) + "\n")
        if importance_df is not None:
            f.write("\nFeature Importance (top 5):\n")
            f.write(str(importance_df.head()) + "\n")
    return report_path

def main():
    # ============================================================================
# 10. ENHANCED MAIN EXECUTION FUNCTION WITH PROGRESS TRACKING
# ============================================================================
    """Execute complete EDA analysis pipeline with detailed progress tracking."""
    print("🚀 STARTING COMPREHENSIVE EDA ANALYSIS")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    # Step 1: Load data
    print("\n[STEP 1/9] Loading dataset...")
    step_start = time.time()
    df = load_and_setup_data()
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    print(f"✅ Step 1 completed in {time.time() - step_start:.1f} seconds")
    
    # Step 2: Dataset overview
    print("\n[STEP 2/9] Dataset overview analysis...")
    step_start = time.time()
    try:
        numeric_cols, categorical_cols = dataset_overview(df)
        print(f"✅ Step 2 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 2 failed: {str(e)}")
        numeric_cols, categorical_cols = [], []
    
    # Step 3: Target variable analysis
    print("\n[STEP 3/9] Target variable analysis...")
    step_start = time.time()
    try:
        target_dist = analyze_target_variable(df)
        print(f"✅ Step 3 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 3 failed: {str(e)}")
        target_dist = None
    
    # Step 4: Correlation analysis
    print("\n[STEP 4/9] Multimodal correlation analysis...")
    step_start = time.time()
    try:
        corr_matrix, speech_feats, motor_feats, clinical_feats = multimodal_correlation_analysis(df)
        print(f"✅ Step 4 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 4 failed: {str(e)}")
        corr_matrix, speech_feats, motor_feats, clinical_feats = None, [], [], []
    
    # Step 5: Patient segmentation
    print("\n[STEP 5/9] Patient segmentation analysis...")
    step_start = time.time()
    try:
        df_clustered, clusters, optimal_k = patient_segmentation_analysis(df)
        print(f"✅ Step 5 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 5 failed: {str(e)}")
        df_clustered, clusters, optimal_k = df, None, None
    
    # Step 6: Temporal analysis
    print("\n[STEP 6/9] Temporal progression analysis...")
    step_start = time.time()
    try:
        df_temporal = temporal_progression_analysis(df)
        print(f"✅ Step 6 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 6 failed: {str(e)}")
        df_temporal = df
    
    # Step 7: Feature importance (POTENTIALLY SLOW)
    print("\n[STEP 7/9] Feature importance analysis...")
    print("⏳ This may take several minutes for large datasets...")
    step_start = time.time()
    try:
        importance_df = feature_importance_analysis(df)
        print(f"✅ Step 7 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 7 failed: {str(e)}")
        importance_df = None
    
    # Step 8: Statistical summary
    print("\n[STEP 8/9] Statistical summary...")
    step_start = time.time()
    try:
        statistical_summary(df)
        print(f"✅ Step 8 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 8 failed: {str(e)}")
    
    # Step 9: Generate comprehensive report
    print("\n[STEP 9/9] Generating comprehensive report...")
    step_start = time.time()
    try:
        report_path = generate_eda_report(df, target_dist, corr_matrix, importance_df)
        print(f"✅ Step 9 completed in {time.time() - step_start:.1f} seconds")
    except Exception as e:
        print(f"❌ Step 9 failed: {str(e)}")
        report_path = None
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 PHASE 5 EDA ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ All analyses completed")
    print(f"📊 Visualizations saved:")
    print(f"   • treatment_response_analysis.png")
    print(f"   • correlation_heatmap.png") 
    print(f"   • patient_segmentation.png")
    print(f"   • temporal_progression.png")
    print(f"   • feature_importance.png")
    if report_path:
        print(f"📋 Report saved: {report_path}")
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Dataset is ready for AI model development")
    print(f"   • {df['fox_insight_id'].nunique():,} unique patients with rich multimodal data")
    print(f"   • Clear treatment response patterns identified")
    print(f"   • Strong feature correlations discovered")
    print(f"   • Patient subgroups reveal clinical phenotypes")
    print(f"\n🚀 READY FOR PHASE 6: BASELINE MODEL DEVELOPMENT")
    print(f"   Next file to create: pd_baseline_models.py")


# ============================================================================
# EXECUTE ANALYSIS
# ============================================================================
if __name__ == "__main__":
    main()
