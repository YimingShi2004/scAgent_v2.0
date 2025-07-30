#!/usr/bin/env python3
"""
Performance Comparison Script
Compare performance differences before and after optimization
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent_2025'))

def compare_performance():
    """Compare performance before and after optimization"""
    print("📊 Performance Comparison: Before vs After Optimization")
    print("=" * 80)
    
    # Test configuration
    test_samples = 1000
    
    results = {
        "before_optimization": {},
        "after_optimization": {}
    }
    
    # Test optimized performance
    print(f"\n🚀 Testing Optimized Version (After)")
    print("-" * 50)
    
    try:
        from enhanced_sc_eqtl_analysis import EnhancedScEqtlAnalysis
        
        start_time = time.time()
        
        analysis = EnhancedScEqtlAnalysis(
            max_ai_workers=64,
            enable_gpu=True,
            include_download_urls=True
        )
        
        result = analysis.run_enhanced_analysis(
            max_samples=test_samples,
            batch_size=20000,
            output_file=f"optimized_test_{test_samples}.csv",
            enable_ai=True,
            gpu_batch_size=1000,
            auto_download=False
        )
        
        end_time = time.time()
        
        if result["status"] == "completed":
            stats = result["processing_statistics"]
            results["after_optimization"] = {
                "total_time": stats["total_time"],
                "filter_time": stats["filter_time"],
                "analysis_time": stats["analysis_time"],
                "records_per_second": stats["records_per_second"],
                "total_processed": stats["total_processed"],
                "hard_filtered": stats["hard_filtered"],
                "final_passed": stats["final_passed"],
                "pass_rate": (stats["final_passed"] / stats["total_processed"] * 100) if stats["total_processed"] > 0 else 0,
                "wall_clock_time": end_time - start_time
            }
            
            print(f"✅ Optimized version completed!")
            print(f"   Total Time: {stats['total_time']:.2f}s")
            print(f"   Records/Second: {stats['records_per_second']:.1f}")
            print(f"   Pass Rate: {results['after_optimization']['pass_rate']:.2f}%")
            print(f"   Wall Clock Time: {results['after_optimization']['wall_clock_time']:.2f}s")
        else:
            print(f"❌ Optimized version failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Optimized test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    
    if results["after_optimization"]:
        after = results["after_optimization"]
        
        print(f"\n🎯 OPTIMIZED VERSION RESULTS:")
        print(f"   Total Processed: {after['total_processed']:,}")
        print(f"   Hard Filtered: {after['hard_filtered']:,}")
        print(f"   Final Passed: {after['final_passed']:,}")
        print(f"   Pass Rate: {after['pass_rate']:.2f}%")
        print(f"   Filter Time: {after['filter_time']:.2f}s")
        print(f"   Analysis Time: {after['analysis_time']:.2f}s")
        print(f"   Total Time: {after['total_time']:.2f}s")
        print(f"   Records/Second: {after['records_per_second']:.1f}")
        print(f"   Wall Clock Time: {after['wall_clock_time']:.2f}s")
        
        # Calculate efficiency metrics
        efficiency = after['final_passed'] / after['total_time'] if after['total_time'] > 0 else 0
        print(f"   Efficiency (passed samples/second): {efficiency:.2f}")
        
        # Save detailed results
        save_detailed_results(results, test_samples)
    
    print("\n" + "=" * 80)
    print("🎉 Performance comparison completed!")
    print("=" * 80)

def save_detailed_results(results, test_samples):
    """Save detailed results to CSV"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_comparison_{test_samples}_{timestamp}.csv"
        
        # Prepare data
        data = []
        for version, metrics in results.items():
            if metrics:  # Only save versions with data
                row = {
                    "version": version,
                    "test_samples": test_samples,
                    "timestamp": timestamp,
                    **metrics
                }
                data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n📁 Detailed results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save detailed results: {e}")

def analyze_output_quality():
    """Analyze output quality"""
    print("\n🔍 Analyzing Output Quality")
    print("-" * 50)
    
    try:
        # Find latest output files
        import glob
        csv_files = glob.glob("optimized_test_*.csv")
        
        if not csv_files:
            print("❌ No output files found for analysis")
            return
        
        # Use latest file
        latest_file = max(csv_files, key=os.path.getctime)
        print(f"📊 Analyzing: {latest_file}")
        
        df = pd.read_csv(latest_file)
        
        print(f"   Total Records: {len(df):,}")
        print(f"   Unique SRA IDs: {df['sra_id'].nunique():,}")
        print(f"   Experiment Types:")
        if 'experiment_type' in df.columns:
            type_counts = df['experiment_type'].value_counts()
            for exp_type, count in type_counts.items():
                print(f"     {exp_type}: {count:,}")
        
        print(f"   Tumor Samples: {df['is_tumor'].sum() if 'is_tumor' in df.columns else 'N/A'}")
        print(f"   Age Information Available: {df['age_value'].notna().sum() if 'age_value' in df.columns else 'N/A'}")
        print(f"   PMID Available: {df['pmid'].notna().sum() if 'pmid' in df.columns else 'N/A'}")
        
        # Check SRA Lite URLs
        if 'sra_lite_url' in df.columns:
            valid_urls = df['sra_lite_url'].str.contains('trace.ncbi.nlm.nih.gov').sum()
            print(f"   Valid SRA Lite URLs: {valid_urls:,}")
        
    except Exception as e:
        print(f"❌ Output analysis failed: {e}")

def main():
    """Main function"""
    print("📊 Performance Comparison Suite")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run performance comparison
    compare_performance()
    
    # Analyze output quality
    analyze_output_quality()
    
    print("\n" + "=" * 80)
    print("🎉 Performance analysis completed!")
    print("=" * 80)

if __name__ == "__main__":
    main() 