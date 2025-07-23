"""
Fox Insight Multimodal Data Integration - MEMORY OPTIMIZED
========================================================
Handles large datasets with smart sampling and chunked processing.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
import gc  # Garbage collection for memory management
warnings.filterwarnings('ignore')

# ============================================================================
# MEMORY-OPTIMIZED CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"C:\Users\niket\OneDrive\Desktop\filtered_data")

# Focused on CORE columns for multimodal AI
CORE_MULTIMODAL_CONFIG = {
    
    # ================================
    # 1. PATIENT BASELINE (Essential)
    # ================================
    "Filtered_About.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "Sex", "Education", "Income", "Employment"
        ],
        "sample_size": 50000,  # Limit to 50k patients
        "description": "Core demographics"
    },

    # ================================
    # 2. PD CLINICAL CORE
    # ================================
    "filtered_Clinical_Global_Impression_of_Change_PD.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed", "CGIPD"
        ],
        "sample_size": 30000,
        "description": "Clinical Global Impression"
    },

    # ================================
    # 3. MOTOR COMPENSATION (Critical for AI)
    # ================================
    "filtered_Compensation_Strategies.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "CompDiffWalk", "CompFreezeFreq", 
            "CompExCueEffectTalk", "CompInCueEffectTalk",
            "CompBalanceEffectADL", "CompMotorEffectTalk"
        ],
        "sample_size": 8000,  # Full dataset (it's small)
        "description": "Speech & motor compensation"
    },

    # ================================
    # 4. ON/OFF EPISODES (Critical)
    # ================================
    "filtered_Episodes.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "Episode", "EpisodeOff", "EpisodeOffDur",
            "EpisodeOffWalk", "EpisodeOffAnxious", "EpisodeON"
        ],
        "sample_size": 12000,
        "description": "ON/OFF motor episodes"
    },

    # ================================
    # 5. SYMPTOM PROGRESSION (Core timepoints)
    # ================================
    "filtered_PDPROP.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            # Focus on speech, motor, and mood - core for multimodal
            "PdpropBradySpeech_1", "PdpropPostinGait_1", "PdpropMoodAnxiety_1",
            "PdpropSev_1", "PdpropSev_2", "PdpropSev_3"
        ],
        "sample_size": 20000,
        "description": "Core symptom progression"
    },

    # ================================
    # 6. PHYSICAL ACTIVITY (Functional outcome)
    # ================================
    "filtered_PASE.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "WalkDay", "WalkHours", "Work", "WorkActive"
        ],
        "sample_size": 25000,
        "description": "Physical activity levels"
    },

    # ================================
    # 7. HANDEDNESS (Motor asymmetry)
    # ================================
    "filtered_Handedness.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "HandsSpoon", "HandsWriting"
        ],
        "sample_size": 15000,
        "description": "Motor asymmetry"
    },

    # ================================
    # 8. CLINICAL ASSESSMENTS (FIVE Study)
    # ================================
    "filtered_FIVE.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "FIVEPDVoice", "FIVEPDVoiceAge",  # SPEECH - Critical
            "FIVEPDBalance", "FIVEPDShake",
            "FIVEUPDRS2_1", "FIVEUPDRS2_2", "FIVEMOCAScore1"
        ],
        "sample_size": 185,  # Use all (small dataset)
        "description": "Clinical assessments with speech"
    },

    # ================================
    # 9. MOOD & MENTAL HEALTH
    # ================================
    "filtered_Mood.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "MindStress", "MindAnxietyPDSympTremor", "MindTechMindful"
        ],
        "sample_size": 15000,
        "description": "Mental health status"
    },

    # ================================
    # 10. LIFESTYLE FACTORS
    # ================================
    "filtered_EEQ_alcohol.csv": {
        "columns": [
            "fox_insight_id", "age", "days_elapsed",
            "alq1", "al6_beer"
        ],
        "sample_size": 2480,
        "description": "Alcohol use"
    }
}

# ============================================================================
# MEMORY-EFFICIENT FUNCTIONS
# ============================================================================

def read_dataset_sampled(file_path: Path, columns: List[str], sample_size: int, description: str) -> Optional[pd.DataFrame]:
    """Read dataset with intelligent sampling for memory efficiency."""
    try:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path.name}")
            return None
        
        # First check actual columns
        df_sample = pd.read_csv(file_path, nrows=0)
        actual_columns = df_sample.columns.tolist()
        
        # Find available columns
        available_columns = [col for col in columns if col in actual_columns]
        missing_columns = [col for col in columns if col not in actual_columns]
        
        if missing_columns:
            print(f"   ⚠️  Missing: {missing_columns[:3]}...")
        
        if not available_columns:
            print(f"   ❌ No matching columns in {file_path.name}")
            return None
        
        # Memory-efficient reading with sampling
        print(f"   🔄 Reading {file_path.name}...")
        
        # Read in chunks and sample
        chunk_size = 10000
        sampled_chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, usecols=available_columns, chunksize=chunk_size, low_memory=False):
            total_rows += len(chunk)
            
            # Sample from this chunk
            if len(chunk) > 0:
                sample_size_chunk = min(sample_size // 10, len(chunk))  # Take portion of sample from each chunk
                if sample_size_chunk > 0:
                    sampled_chunk = chunk.sample(n=sample_size_chunk, random_state=42)
                    sampled_chunks.append(sampled_chunk)
            
            # Stop if we have enough
            if sum(len(chunk) for chunk in sampled_chunks) >= sample_size:
                break
        
        if not sampled_chunks:
            print(f"   ❌ No data found in {file_path.name}")
            return None
        
        # Combine chunks
        df = pd.concat(sampled_chunks, ignore_index=True)
        
        # Final sampling if still too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"   ✅ {file_path.name}: {len(df):,} rows (sampled from {total_rows:,}), {len(available_columns)} columns")
        print(f"      📝 {description}")
        
        # Memory cleanup
        del sampled_chunks
        gc.collect()
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error reading {file_path.name}: {str(e)}")
        return None

def create_temporal_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create a temporal key for better merging."""
    if 'age' in df.columns and 'days_elapsed' in df.columns:
        # Create age bins for better matching
        df['age_bin'] = pd.cut(df['age'], bins=10, labels=False)
        df['temporal_key'] = df['fox_insight_id'].astype(str) + '_' + df['age_bin'].astype(str)
    else:
        df['temporal_key'] = df['fox_insight_id'].astype(str)
    return df

def merge_datasets_smart(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Smart merge that handles temporal relationships efficiently."""
    if not datasets:
        raise ValueError("No datasets to merge")
    
    print(f"\n🔄 Smart merging {len(datasets)} datasets...")
    
    # Start with demographics as base
    base_name = "Filtered_About.csv"
    if base_name not in datasets:
        base_name = list(datasets.keys())[0]
    
    master_df = datasets[base_name].copy()
    master_df = create_temporal_key(master_df)
    
    print(f"   📊 Base ({base_name}): {len(master_df):,} rows, {len(master_df.columns)} columns")
    
    # Get unique patients for baseline merging
    unique_patients = master_df['fox_insight_id'].unique()
    print(f"   👥 Unique patients in base: {len(unique_patients):,}")
    
    # Merge other datasets
    for name, df in datasets.items():
        if name == base_name:
            continue
            
        print(f"   🔄 Merging {name}...")
        
        # Filter to patients we have in base
        df_filtered = df[df['fox_insight_id'].isin(unique_patients)].copy()
        
        if len(df_filtered) == 0:
            print(f"      ⚠️  No overlapping patients with {name}")
            continue
        
        # Create temporal key
        df_filtered = create_temporal_key(df_filtered)
        
        # For each patient, take the most recent record
        df_latest = df_filtered.groupby('fox_insight_id').apply(
            lambda x: x.loc[x['age'].idxmax()] if 'age' in x.columns else x.iloc[0]
        ).reset_index(drop=True)
        
        before_cols = len(master_df.columns)
        
        # Merge on fox_insight_id only (latest records)
        master_df = master_df.merge(
            df_latest.drop(['temporal_key'], axis=1, errors='ignore'), 
            on='fox_insight_id', 
            how='left',
            suffixes=('', f'_{name.split(".")[0]}')
        )
        
        after_cols = len(master_df.columns)
        overlap = len(df_latest)
        
        print(f"      ✅ Added {after_cols - before_cols} columns, {overlap:,} patient matches")
        
        # Memory cleanup
        del df_filtered, df_latest
        gc.collect()
    
    # Remove temporal key
    master_df = master_df.drop(['temporal_key'], axis=1, errors='ignore')
    
    return master_df

def create_analysis_summary(df: pd.DataFrame) -> None:
    """Create summary for data analysis."""
    print(f"\n📊 INTEGRATED DATASET SUMMARY")
    print("-" * 80)
    print(f"   🔢 Total rows: {len(df):,}")
    print(f"   📋 Total columns: {len(df.columns):,}")
    print(f"   👥 Unique patients: {df['fox_insight_id'].nunique():,}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   💾 Memory usage: {memory_mb:.1f} MB")
    
    # Completeness by category
    print(f"\n🎯 KEY COLUMNS COMPLETENESS:")
    key_cols = [
        'fox_insight_id', 'age', 'Sex', 'Education',
        'CGIPD', 'CompDiffWalk', 'Episode', 'PdpropBradySpeech_1',
        'WalkDay', 'FIVEPDVoice', 'MindStress'
    ]
    
    for col in key_cols:
        if col in df.columns:
            completeness = (df[col].count() / len(df)) * 100
            print(f"   {col:<25} {completeness:>6.1f}%")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Memory-optimized main execution."""
    print("=" * 80)
    print("🧠 FOX INSIGHT MULTIMODAL DATA INTEGRATION - MEMORY OPTIMIZED")
    print("   Smart Sampling for Large Dataset Processing")
    print("=" * 80)
    
    if not DATA_DIR.exists():
        print(f"❌ ERROR: Data directory not found: {DATA_DIR}")
        return
    
    print(f"✅ Data directory found: {DATA_DIR}")
    print(f"📊 Processing {len(CORE_MULTIMODAL_CONFIG)} core datasets")
    print(f"🎯 Focus: Multimodal AI features (Speech, Motor, Cognitive)")
    
    # Read datasets with sampling
    datasets = {}
    print(f"\n📁 Reading sampled datasets...")
    print("-" * 80)
    
    for filename, config in CORE_MULTIMODAL_CONFIG.items():
        print(f"\n🔍 Processing: {filename}")
        file_path = DATA_DIR / filename
        df = read_dataset_sampled(
            file_path, 
            config["columns"], 
            config["sample_size"], 
            config["description"]
        )
        
        if df is not None:
            datasets[filename] = df
    
    if not datasets:
        print("\n❌ No datasets successfully loaded!")
        return
    
    print(f"\n✅ Successfully loaded {len(datasets)} datasets")
    
    # Smart merge
    print("-" * 80)
    master_df = merge_datasets_smart(datasets)
    
    # Analysis summary
    create_analysis_summary(master_df)
    
    # Save results
    output_path = DATA_DIR / "fox_insight_multimodal_CORE.csv"
    master_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path.name}")
    
    # Create feature importance ranking
    feature_summary = pd.DataFrame({
        'Column': master_df.columns,
        'Non_Null_Count': master_df.count(),
        'Completeness_%': (master_df.count() / len(master_df) * 100).round(1),
        'Unique_Values': master_df.nunique()
    }).sort_values('Completeness_%', ascending=False)
    
    summary_path = DATA_DIR / "feature_summary_CORE.csv"
    feature_summary.to_csv(summary_path, index=False)
    
    print(f"\n🏆 TOP 15 FEATURES FOR MULTIMODAL AI:")
    print("-" * 80)
    for _, row in feature_summary.head(15).iterrows():
        print(f"   {row['Column']:<30} {row['Completeness_%']:>6.1f}% ({row['Unique_Values']:>4} unique)")
    
    print(f"\n✅ MEMORY-OPTIMIZED INTEGRATION COMPLETE!")
    print(f"   📊 Dataset ready for multimodal AI pipeline")
    print(f"   🎯 Focus on speech, motor, and cognitive features")
    print(f"   💾 Manageable size: {master_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
