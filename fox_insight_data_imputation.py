"""
Fox Insight Dataset - Missing Data Imputation & Completeness Enhancement
======================================================================
Author: Niket
Purpose: Improve data completeness for multimodal PD treatment monitoring
Input: fox_insight_multimodal_CORE.csv
Output: fox_insight_multimodal_CORE_IMPUTED.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIXED IMPORTS - CORRECT ORDER
# ============================================================================

# Enable experimental features FIRST
from sklearn.experimental import enable_iterative_imputer  # Must be BEFORE IterativeImputer

# Now import the imputers
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# LOAD YOUR INTEGRATED DATASET
# ============================================================================

# Load the dataset you created
DATA_PATH = r"C:\Users\niket\OneDrive\Desktop\filtered_data\fox_insight_multimodal_CORE.csv"
df = pd.read_csv(DATA_PATH)

print("🔍 ORIGINAL DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique patients: {df['fox_insight_id'].nunique():,}")

# ============================================================================
# ANALYZE MISSING DATA PATTERNS
# ============================================================================

def analyze_completeness(df):
    """Analyze missing data patterns."""
    completeness = pd.DataFrame({
        'Column': df.columns,
        'Non_Null': df.count(),
        'Missing': df.isnull().sum(),
        'Completeness_%': (df.count() / len(df) * 100).round(1),
        'Data_Type': df.dtypes
    }).sort_values('Completeness_%', ascending=False)
    
    return completeness

completeness_report = analyze_completeness(df)
print(f"\n📊 COMPLETENESS ANALYSIS")
print("=" * 60)
print(completeness_report.head(20).to_string(index=False))

# ============================================================================
# CATEGORIZE COLUMNS BY COMPLETENESS LEVEL
# ============================================================================

def categorize_columns_by_completeness(completeness_df):
    """Categorize columns by completeness level."""
    high_complete = completeness_df[completeness_df['Completeness_%'] >= 70]['Column'].tolist()
    medium_complete = completeness_df[
        (completeness_df['Completeness_%'] >= 20) & 
        (completeness_df['Completeness_%'] < 70)
    ]['Column'].tolist()
    low_complete = completeness_df[
        (completeness_df['Completeness_%'] >= 1) & 
        (completeness_df['Completeness_%'] < 20)
    ]['Column'].tolist()
    very_low_complete = completeness_df[completeness_df['Completeness_%'] < 1]['Column'].tolist()
    
    return high_complete, medium_complete, low_complete, very_low_complete

high_cols, medium_cols, low_cols, very_low_cols = categorize_columns_by_completeness(completeness_report)

print(f"\n🎯 COLUMN CATEGORIZATION")
print("=" * 60)
print(f"HIGH completeness (≥70%): {len(high_cols)} columns")
print(f"MEDIUM completeness (20-70%): {len(medium_cols)} columns")
print(f"LOW completeness (1-20%): {len(low_cols)} columns")
print(f"VERY LOW completeness (<1%): {len(very_low_cols)} columns")

# ============================================================================
# STRATEGY 1: DROP EXTREMELY SPARSE COLUMNS
# ============================================================================

def handle_very_low_completeness(df, very_low_cols, threshold=0.5):
    """Drop or flag columns with extremely low completeness."""
    print(f"\n⚠️  HANDLING VERY LOW COMPLETENESS COLUMNS")
    print("=" * 60)
    
    to_drop = []
    to_keep = []
    
    for col in very_low_cols:
        if col in ['fox_insight_id']:  # Never drop ID
            continue
            
        completeness = df[col].count() / len(df) * 100
        
        # Keep if clinically critical (speech/motor biomarkers)
        if any(keyword in col.lower() for keyword in ['speech', 'voice', 'talk', 'bradyspeech']):
            to_keep.append(col)
            print(f"KEEP: {col} ({completeness:.1f}%) - Critical speech biomarker")
        else:
            to_drop.append(col)
            print(f"DROP: {col} ({completeness:.1f}%) - Too sparse")
    
    return to_drop, to_keep

drop_cols, keep_very_low = handle_very_low_completeness(df, very_low_cols)

# Drop extremely sparse non-critical columns
df_cleaned = df.drop(columns=drop_cols)
print(f"\n✅ Dropped {len(drop_cols)} extremely sparse columns")
print(f"📊 New dataset: {len(df_cleaned)} rows × {len(df_cleaned.columns)} columns")

# ============================================================================
# STRATEGY 2: CREATE DOMAIN-SPECIFIC IMPUTATION GROUPS
# ============================================================================

def create_imputation_groups():
    """Define related features for group-wise imputation."""
    return {
        'demographics': ['age', 'Sex', 'Education', 'Income', 'Employment'],
        'motor_function': ['CompDiffWalk', 'Episode', 'EpisodeOff', 'WalkDay', 'Work', 'HandsWriting'],
        'speech_communication': ['PdpropBradySpeech_1', 'FIVEPDVoice', 'CompExCueEffectTalk'],
        'clinical_scores': ['CGIPD', 'PdpropSev_1', 'PdpropSev_2', 'PdpropSev_3'],
        'activity_lifestyle': ['WalkDay', 'Work', 'LeisureDay', 'alq1']
    }

imputation_groups = create_imputation_groups()

# ============================================================================
# STRATEGY 3: SIMPLIFIED IMPUTATION METHODS (AVOIDING EXPERIMENTAL FEATURES)
# ============================================================================

def impute_by_group(df, group_name, columns, method='knn'):
    """Impute missing values within a feature group."""
    print(f"\n🔄 Imputing {group_name} features using {method.upper()}")
    
    # Filter to existing columns
    existing_cols = [col for col in columns if col in df.columns]
    if not existing_cols:
        print(f"   ⚠️  No columns found for {group_name}")
        return df
    
    print(f"   📋 Columns: {existing_cols}")
    
    # Prepare data
    group_data = df[existing_cols].copy()
    
    # Handle categorical variables
    categorical_cols = group_data.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        mask = group_data[col].notna()
        if mask.sum() > 0:  # Only if there's data to encode
            group_data.loc[mask, col] = le.fit_transform(group_data.loc[mask, col])
            label_encoders[col] = le
    
    # Convert to numeric
    group_data = group_data.apply(pd.to_numeric, errors='coerce')
    
    # Apply imputation
    if method == 'knn':
        try:
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = pd.DataFrame(
                imputer.fit_transform(group_data),
                columns=existing_cols,
                index=group_data.index
            )
        except Exception as e:
            print(f"   ⚠️  KNN failed: {str(e)}")
            print(f"   🔄 Using median imputation instead")
            imputed_data = group_data.fillna(group_data.median())
    
    elif method == 'median':
        imputed_data = group_data.fillna(group_data.median())
    
    else:  # Default to median
        imputed_data = group_data.fillna(group_data.median())
    
    # Decode categorical variables back
    for col, le in label_encoders.items():
        if col in imputed_data.columns:
            imputed_data[col] = imputed_data[col].round().astype(int)
            # Ensure values are within valid range
            imputed_data[col] = np.clip(imputed_data[col], 0, len(le.classes_) - 1)
            imputed_data[col] = le.inverse_transform(imputed_data[col])
    
    # Update original dataframe
    df[existing_cols] = imputed_data
    
    print(f"   ✅ Imputation complete")
    
    return df

# ============================================================================
# STRATEGY 4: TEMPORAL IMPUTATION FOR LONGITUDINAL DATA
# ============================================================================

def temporal_imputation(df, patient_id_col='fox_insight_id', time_col='age'):
    """Forward-fill and backward-fill within patients over time."""
    print(f"\n⏰ TEMPORAL IMPUTATION")
    print("=" * 40)
    
    # Sort by patient and time
    df_temporal = df.sort_values([patient_id_col, time_col]).copy()
    
    # Identify longitudinal columns (severity scores, symptoms)
    longitudinal_cols = [col for col in df.columns if any(
        keyword in col.lower() for keyword in ['sev_', 'prop', 'episode', 'comp']
    )]
    
    print(f"   📊 Processing {len(longitudinal_cols)} longitudinal features")
    
    # Forward fill and backward fill within each patient
    for col in longitudinal_cols:
        if col in df_temporal.columns:
            df_temporal[col] = df_temporal.groupby(patient_id_col)[col].ffill()
            df_temporal[col] = df_temporal.groupby(patient_id_col)[col].bfill()
    
    return df_temporal.sort_index()  # Restore original order

# ============================================================================
# APPLY ALL IMPUTATION STRATEGIES
# ============================================================================

def comprehensive_imputation(df):
    """Apply all imputation strategies systematically."""
    print(f"\n🚀 COMPREHENSIVE IMPUTATION PIPELINE")
    print("=" * 60)
    
    df_imputed = df.copy()
    
    # Step 1: Temporal imputation for longitudinal data
    df_imputed = temporal_imputation(df_imputed)
    
    # Step 2: Group-wise imputation (using KNN, fallback to median)
    for group_name, columns in imputation_groups.items():
        df_imputed = impute_by_group(df_imputed, group_name, columns, method='knn')
    
    # Step 3: Handle remaining missing values with simple strategies
    print(f"\n🔧 FINAL CLEANUP")
    print("=" * 30)
    
    # Numeric columns: median imputation
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_imputed[col].isnull().sum() > 0:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
            print(f"   📊 {col}: filled with median ({median_val:.2f})")
    
    # Categorical columns: mode imputation
    categorical_cols = df_imputed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_val = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 'Unknown'
            df_imputed[col] = df_imputed[col].fillna(mode_val)
            print(f"   📝 {col}: filled with mode ({mode_val})")
    
    return df_imputed

# Apply comprehensive imputation
df_final = comprehensive_imputation(df_cleaned)

# ============================================================================
# EVALUATE IMPUTATION RESULTS
# ============================================================================

def evaluate_imputation_results(df_original, df_imputed):
    """Compare before and after imputation."""
    print(f"\n📈 IMPUTATION RESULTS")
    print("=" * 60)
    
    original_completeness = analyze_completeness(df_original)
    final_completeness = analyze_completeness(df_imputed)
    
    comparison = pd.merge(
        original_completeness[['Column', 'Completeness_%']],
        final_completeness[['Column', 'Completeness_%']],
        on='Column',
        suffixes=('_before', '_after')
    )
    comparison['Improvement'] = comparison['Completeness_%_after'] - comparison['Completeness_%_before']
    
    # Show biggest improvements
    improved = comparison[comparison['Improvement'] > 0].sort_values('Improvement', ascending=False)
    
    print(f"🎯 BIGGEST COMPLETENESS IMPROVEMENTS:")
    print("-" * 50)
    for _, row in improved.head(10).iterrows():
        print(f"{row['Column']:<30} {row['Completeness_%_before']:>6.1f}% → {row['Completeness_%_after']:>6.1f}% (+{row['Improvement']:>5.1f}%)")
    
    # Overall summary
    print(f"\n📊 OVERALL SUMMARY:")
    print("-" * 30)
    print(f"Columns with 100% completeness: {(final_completeness['Completeness_%'] == 100).sum()}")
    print(f"Columns with >90% completeness: {(final_completeness['Completeness_%'] > 90).sum()}")
    print(f"Columns with >50% completeness: {(final_completeness['Completeness_%'] > 50).sum()}")
    
    return comparison

results_comparison = evaluate_imputation_results(df_cleaned, df_final)

# ============================================================================
# SAVE IMPROVED DATASET
# ============================================================================

# Save the imputed dataset
output_path = DATA_PATH.replace('.csv', '_IMPUTED.csv')
df_final.to_csv(output_path, index=False)

print(f"\n💾 SAVING IMPROVED DATASET")
print("=" * 40)
print(f"Original file: {DATA_PATH}")
print(f"Improved file: {output_path}")
print(f"Size: {df_final.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

print(f"\n✅ IMPUTATION COMPLETE!")
print("=" * 40)
print(f"🎯 Your dataset is now ready for AI model training")
print(f"📊 Improved completeness across all features")
print(f"🚀 Next step: Start building your multimodal PD models!")
