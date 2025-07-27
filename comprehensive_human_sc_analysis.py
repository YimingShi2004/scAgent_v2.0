#!/usr/bin/env python3
"""
Comprehensive Human Single-Cell Analysis for sc-eQTL Dataset Discovery
Complete batch processing with CSV output for all human samples
"""

import sys
import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import time

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent'))

from scAgent.utils_sra_geo_ft2_optimizer import SraGeoFt2Optimizer
from scAgent.db.connect import get_connection
import psycopg2.extras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveHumanScAnalysis:
    """
    Comprehensive analysis system for human single-cell datasets suitable for sc-eQTL analysis.
    """
    
    def __init__(self, max_ai_workers: int = 15):
        self.optimizer = SraGeoFt2Optimizer()
        self.results = []
        self.stats = {
            "total_processed": 0,
            "human_samples": 0,
            "passed_all_filters": 0,
            "single_cell_identified": 0,
            "cell_lines_excluded": 0,
            "ai_assisted_decisions": 0,
            "processing_errors": 0
        }
        self.max_ai_workers = max_ai_workers  # Maximum concurrent AI calls
        self.stats_lock = threading.Lock()  # Thread-safe statistics updates
        
    def find_all_human_samples(self, batch_size: int = 10000, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find all human samples in the database using optimized queries.
        
        Args:
            batch_size: Number of records per batch
            limit: Maximum records to process (None for all)
            
        Returns:
            List of human sample records
        """
        logger.info("Starting comprehensive human sample discovery...")
        
        try:
            conn = get_connection()
            
            # First, get count of potential human samples
            count_query = """
            SELECT COUNT(*) as total_count
            FROM "merged"."sra_geo_ft2"
            WHERE 
                LOWER(COALESCE("organism_ch1", '')) LIKE %s OR
                LOWER(COALESCE("scientific_name", '')) LIKE %s OR
                LOWER(COALESCE("experiment_title", '')) LIKE %s OR
                LOWER(COALESCE("study_title", '')) LIKE %s OR
                LOWER(COALESCE("summary", '')) LIKE %s
            """
            
            with conn.cursor() as cur:
                cur.execute(count_query, ('%homo sapiens%', '%homo sapiens%', '%homo sapiens%', '%human%', '%human%'))
                total_potential = cur.fetchone()[0]
                logger.info(f"Found {total_potential:,} potential human samples in database")
            
            # Process limit
            if limit:
                process_count = min(limit, total_potential)
            else:
                process_count = total_potential
                
            logger.info(f"Will process {process_count:,} samples")
            
            # Fetch samples in batches with progress bar
            all_samples = []
            offset = 0
            
            main_query = """
            SELECT * FROM "merged"."sra_geo_ft2"
            WHERE 
                LOWER(COALESCE("organism_ch1", '')) LIKE %s OR
                LOWER(COALESCE("scientific_name", '')) LIKE %s OR
                LOWER(COALESCE("experiment_title", '')) LIKE %s OR
                LOWER(COALESCE("study_title", '')) LIKE %s OR
                LOWER(COALESCE("summary", '')) LIKE %s
            ORDER BY "sra_ID"
            LIMIT %s OFFSET %s
            """
            
            # Progress bar for fetching
            with tqdm(total=process_count, desc="Fetching human samples", unit="samples") as pbar:
                while offset < process_count:
                    current_batch_size = min(batch_size, process_count - offset)
                    
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(main_query, (
                            '%homo sapiens%', '%homo sapiens%', '%homo sapiens%', 
                            '%human%', '%human%', current_batch_size, offset
                        ))
                        batch_samples = cur.fetchall()
                        
                    if not batch_samples:
                        break
                        
                    all_samples.extend([dict(sample) for sample in batch_samples])
                    offset += len(batch_samples)
                    pbar.update(len(batch_samples))
            
            conn.close()
            logger.info(f"Successfully fetched {len(all_samples):,} human sample candidates")
            return all_samples
            
        except Exception as e:
            logger.error(f"Error fetching human samples: {e}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def process_single_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single sample through the filtering pipeline.
        
        Args:
            sample: Sample record
            
        Returns:
            Enhanced result record or None if error
        """
        try:
            # Apply complete filtering pipeline
            result = self.optimizer.filter_record_optimized(sample)
            
            # Enhance result with additional metadata
            enhanced_result = self._enhance_result_record(sample, result)
            
            # Update statistics (thread-safe)
            with self.stats_lock:
                self._update_stats(result)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('sra_ID', 'unknown')}: {e}")
            with self.stats_lock:
                self.stats["processing_errors"] += 1
            return None
    
    def process_samples_batch_parallel(self, samples: List[Dict[str, Any]], enable_ai: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of samples using parallel processing for AI calls.
        
        Args:
            samples: List of sample records
            enable_ai: Enable AI-assisted analysis
            
        Returns:
            List of processed results
        """
        batch_results = []
        
        # Separate samples that need AI from those that don't
        ai_needed_samples = []
        non_ai_samples = []
        
        # Pre-filter to identify which samples need AI
        for sample in samples:
            # Quick check if sample might need AI assistance
            experiment_title = str(sample.get('experiment_title', '')).lower()
            if 'rna-seq' in experiment_title and 'single' not in experiment_title:
                ai_needed_samples.append(sample)
            else:
                non_ai_samples.append(sample)
        
        logger.info(f"Samples requiring AI analysis: {len(ai_needed_samples)}, Regular processing: {len(non_ai_samples)}")
        
        # Process non-AI samples quickly first
        with tqdm(total=len(non_ai_samples), desc="Processing non-AI samples", unit="samples") as pbar:
            for sample in non_ai_samples:
                result = self.process_single_sample(sample)
                if result:
                    batch_results.append(result)
                pbar.update(1)
        
        # Process AI-needed samples in parallel
        if ai_needed_samples and enable_ai:
            with ThreadPoolExecutor(max_workers=self.max_ai_workers) as executor:
                # Submit all AI tasks
                future_to_sample = {
                    executor.submit(self.process_single_sample, sample): sample 
                    for sample in ai_needed_samples
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(ai_needed_samples), desc="Processing AI-assisted samples", unit="samples") as pbar:
                    for future in as_completed(future_to_sample):
                        result = future.result()
                        if result:
                            batch_results.append(result)
                        pbar.update(1)
        
        return batch_results
    
    def _enhance_result_record(self, original_sample: Dict[str, Any], filter_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced result record with complete sc-eQTL information.
        
        Args:
            original_sample: Original database record
            filter_result: Filtering result from optimizer
            
        Returns:
            Enhanced result record for CSV output
        """
        # Base information
        enhanced = {
            # Identifiers
            "sra_ID": original_sample.get('sra_ID', ''),
            "gsm_title": original_sample.get('gsm_title', ''),
            "gse_title": original_sample.get('gse_title', ''),
            "run_alias": original_sample.get('run_alias', ''),
            "experiment_alias": original_sample.get('experiment_alias', ''),
            
            # Organism information
            "organism_ch1": original_sample.get('organism_ch1', ''),
            "scientific_name": original_sample.get('scientific_name', ''),
            "organism": original_sample.get('organism', ''),
            
            # Experiment information
            "experiment_title": original_sample.get('experiment_title', ''),
            "study_title": original_sample.get('study_title', ''),
            "study_abstract": original_sample.get('study_abstract', ''),
            "summary": original_sample.get('summary', ''),
            
            # Sample information
            "characteristics_ch1": original_sample.get('characteristics_ch1', ''),
            "source_name_ch1": original_sample.get('source_name_ch1', ''),
            "gsm_description": original_sample.get('gsm_description', ''),
            
            # Technical information
            "library_strategy": original_sample.get('library_strategy', ''),
            "library_source": original_sample.get('library_source', ''),
            "platform": original_sample.get('platform', ''),
            "instrument_model": original_sample.get('instrument_model', ''),
            "technology": original_sample.get('technology', ''),
            
            # Data metrics
            "spots": original_sample.get('spots', ''),
            "bases": original_sample.get('bases', ''),
            "spot_length": original_sample.get('spot_length', ''),
            
            # Publication information
            "pubmed_id": original_sample.get('pubmed_id', ''),
            "submission_date": original_sample.get('submission_date', ''),
            
            # Filtering results
            "passes_all_filters": filter_result.get('passes_filter', False),
            "final_confidence_score": filter_result.get('confidence_score', 0.0),
            "rejection_reason": filter_result.get('rejection_reason', ''),
            "processing_time_seconds": filter_result.get('processing_time', 0.0),
        }
        
        # Filter step results
        filter_steps = filter_result.get('filter_steps', {})
        for step_name, step_result in filter_steps.items():
            enhanced[f"{step_name}_passed"] = step_result.get('passed', False)
            enhanced[f"{step_name}_reason"] = step_result.get('reason', '')
            enhanced[f"{step_name}_confidence"] = step_result.get('confidence', 0.0)
            
            # Special handling for experiment type
            if step_name == "single_cell_check" and "experiment_type" in step_result:
                enhanced["experiment_type"] = step_result["experiment_type"]
        
        # sc-eQTL criteria (10 key criteria)
        criteria = filter_result.get('extracted_criteria', {})
        eqtl_criteria = {
            "eqtl_organism": criteria.get('organism', 'Not specified'),
            "eqtl_tissue_type": criteria.get('tissue_type', 'Not specified'),
            "eqtl_cell_type": criteria.get('cell_type', 'Not specified'),
            "eqtl_sample_size": criteria.get('sample_size', 'Not specified'),
            "eqtl_sequencing_platform": criteria.get('sequencing_platform', 'Not specified'),
            "eqtl_project_id": criteria.get('project_id', 'Not specified'),
            "eqtl_publication_info": criteria.get('publication_info', 'Not specified'),
            "eqtl_geographic_location": criteria.get('geographic_location', 'Not specified'),
            "eqtl_age_range": criteria.get('age_range', 'Not specified'),
            "eqtl_disease_status": criteria.get('disease_status', 'Not specified')
        }
        enhanced.update(eqtl_criteria)
        
        # AI assistance indicators
        enhanced["ai_assisted"] = filter_result.get('filter_steps', {}).get('single_cell_check', {}).get('ai_assisted', False)
        enhanced["enhanced_fields_used"] = len(filter_result.get('enhanced_fields_used', []))
        
        return enhanced
    
    def _update_stats(self, filter_result: Dict[str, Any]):
        """Update processing statistics (thread-safe)."""
        self.stats["total_processed"] += 1
        
        # Check filter steps
        filter_steps = filter_result.get('filter_steps', {})
        
        if filter_steps.get('human_check', {}).get('passed', False):
            self.stats["human_samples"] += 1
        
        if filter_steps.get('single_cell_check', {}).get('passed', False):
            self.stats["single_cell_identified"] += 1
            
        if not filter_steps.get('cell_line_check', {}).get('passed', True):  # Note: inverted logic
            self.stats["cell_lines_excluded"] += 1
        
        if filter_result.get('passes_filter', False):
            self.stats["passed_all_filters"] += 1
            
        if filter_result.get('filter_steps', {}).get('single_cell_check', {}).get('ai_assisted', False):
            self.stats["ai_assisted_decisions"] += 1
    
    def save_results_to_csv(self, output_file: str):
        """
        Save all results to a comprehensive CSV file.
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving {len(self.results):,} results to {output_file}")
        
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(self.results)
            
            # Save to CSV with progress bar
            with tqdm(total=1, desc="Saving CSV file", unit="file") as pbar:
                df.to_csv(output_file, index=False, encoding='utf-8')
                pbar.update(1)
            
            # Also save a summary statistics file
            stats_file = output_file.replace('.csv', '_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump({
                    "processing_statistics": self.stats,
                    "result_summary": {
                        "total_results": len(self.results),
                        "passed_filters": len([r for r in self.results if r.get('passes_all_filters', False)]),
                        "pass_rate_percentage": (len([r for r in self.results if r.get('passes_all_filters', False)]) / len(self.results) * 100) if self.results else 0,
                        "average_confidence_score": sum([r.get('final_confidence_score', 0) for r in self.results]) / len(self.results) if self.results else 0,
                    },
                    "generated_at": datetime.now().isoformat(),
                    "script_version": "comprehensive_v2.0_parallel"
                }, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            logger.info(f"Statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def run_comprehensive_analysis(self, 
                                  max_samples: Optional[int] = None,
                                  batch_size: int = 5000,
                                  output_file: Optional[str] = None,
                                  enable_ai: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive analysis of all human samples.
        
        Args:
            max_samples: Maximum samples to process (None for all)
            batch_size: Batch size for processing
            output_file: Output CSV file path
            enable_ai: Enable AI-assisted analysis
            
        Returns:
            Analysis summary
        """
        start_time = datetime.now()
        
        if output_file is None:
            output_file = f"comprehensive_human_sc_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Print startup banner
        startup_banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        COMPREHENSIVE HUMAN SINGLE-CELL ANALYSIS FOR sc-eQTL v2.0             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Configuration:                                                               ║
║   • Start Time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}                         ║
║   • Max Samples        : {str(max_samples or 'ALL'):>10}                                  ║
║   • Batch Size         : {batch_size:>10,}                                  ║
║   • AI Assistance      : {('ENABLED' if enable_ai else 'DISABLED'):>10}                                  ║
║   • Max AI Workers     : {self.max_ai_workers:>10}                                  ║
║   • Output File        : {output_file[:40]:<40} ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        logger.info(startup_banner)
        
        try:
            # Step 1: Find all human samples
            logger.info("\n--- STEP 1: DISCOVERING HUMAN SAMPLES ---")
            all_samples = self.find_all_human_samples(batch_size=batch_size, limit=max_samples)
            
            if not all_samples:
                logger.error("No human samples found. Exiting.")
                return {"status": "failed", "error": "No human samples found"}
            
            logger.info(f"Found {len(all_samples):,} human sample candidates")
            
            # Step 2: Process all samples
            logger.info("\n--- STEP 2: PROCESSING SAMPLES (PARALLEL MODE) ---")
            
            # Process in batches to manage memory
            processing_batch_size = min(batch_size, 1000)  # Smaller batches for processing
            total_batches = (len(all_samples) + processing_batch_size - 1) // processing_batch_size
            
            # Main processing loop with overall progress
            overall_pbar = tqdm(total=len(all_samples), desc="Overall progress", unit="samples", position=0)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * processing_batch_size
                end_idx = min(start_idx + processing_batch_size, len(all_samples))
                batch_samples = all_samples[start_idx:end_idx]
                
                logger.info(f"\nProcessing batch {batch_num + 1}/{total_batches}: samples {start_idx:,}-{end_idx:,}")
                
                # Process batch with parallel AI calls
                batch_results = self.process_samples_batch_parallel(batch_samples, enable_ai=enable_ai)
                self.results.extend(batch_results)
                
                # Update overall progress
                overall_pbar.update(len(batch_samples))
                
                # Periodic statistics report
                if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                    self._report_progress_stats()
            
            overall_pbar.close()
            
            # Step 3: Save results
            logger.info("\n--- STEP 3: SAVING RESULTS ---")
            self.save_results_to_csv(output_file)
            
            # Step 4: Final summary
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            final_summary = {
                "status": "completed",
                "processing_statistics": self.stats.copy(),
                "timing": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_time_seconds": total_time,
                    "samples_per_second": self.stats["total_processed"] / total_time if total_time > 0 else 0
                },
                "results": {
                    "total_results": len(self.results),
                    "output_file": output_file,
                    "passed_all_filters": self.stats["passed_all_filters"],
                    "overall_pass_rate": (self.stats["passed_all_filters"] / self.stats["total_processed"] * 100) if self.stats["total_processed"] > 0 else 0
                }
            }
            
            self._print_final_summary(final_summary)
            return final_summary
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _report_progress_stats(self):
        """Report current progress statistics."""
        stats = self.stats
        
        # Create a formatted progress report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PROGRESS STATISTICS                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Processed        : {stats['total_processed']:>10,} samples                             ║
║ Human Samples          : {stats['human_samples']:>10,} confirmed                           ║
║ Single-Cell Identified : {stats['single_cell_identified']:>10,} experiments                      ║
║ Cell Lines Excluded    : {stats['cell_lines_excluded']:>10,} samples                             ║
║ Passed All Filters     : {stats['passed_all_filters']:>10,} datasets                            ║
║ AI-Assisted Decisions  : {stats['ai_assisted_decisions']:>10,} cases                               ║
║ Processing Errors      : {stats['processing_errors']:>10,} errors                              ║
"""
        
        if stats['total_processed'] > 0:
            pass_rate = (stats['passed_all_filters'] / stats['total_processed']) * 100
            sc_rate = (stats['single_cell_identified'] / stats['human_samples']) * 100 if stats['human_samples'] > 0 else 0
            report += f"""╠══════════════════════════════════════════════════════════════════════════════╣
║ Overall Pass Rate      : {pass_rate:>9.2f}% of all samples                         ║
║ Single-Cell Rate       : {sc_rate:>9.2f}% of human samples                       ║
"""
        
        report += "╚══════════════════════════════════════════════════════════════════════════════╝"
        
        logger.info(report)
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final analysis summary."""
        stats = summary["processing_statistics"]
        timing = summary["timing"]
        results = summary["results"]
        
        # Calculate additional metrics
        hours = timing['total_time_seconds'] / 3600
        minutes = (timing['total_time_seconds'] % 3600) / 60
        seconds = timing['total_time_seconds'] % 60
        
        # Build the summary report
        final_report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE HUMAN SINGLE-CELL ANALYSIS COMPLETED                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────── PROCESSING PERFORMANCE ──────────────────────┐
│ Total Processing Time  : {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s                             │
│ Processing Rate        : {timing['samples_per_second']:>8.1f} samples/second                    │
│ Start Time             : {timing['start_time'][:19]}                      │
│ End Time               : {timing['end_time'][:19]}                        │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────── SAMPLE STATISTICS ───────────────────────────┐
│ Total Processed        : {stats['total_processed']:>10,} samples                      │
│ Human Samples          : {stats['human_samples']:>10,} confirmed ({(stats['human_samples']/stats['total_processed']*100 if stats['total_processed'] > 0 else 0):>5.1f}%)           │
│ Single-Cell Identified : {stats['single_cell_identified']:>10,} experiments ({(stats['single_cell_identified']/stats['human_samples']*100 if stats['human_samples'] > 0 else 0):>5.1f}%)         │
│ Cell Lines Excluded    : {stats['cell_lines_excluded']:>10,} samples                      │
│ AI-Assisted Decisions  : {stats['ai_assisted_decisions']:>10,} cases ({(stats['ai_assisted_decisions']/stats['total_processed']*100 if stats['total_processed'] > 0 else 0):>5.1f}%)              │
│ Processing Errors      : {stats['processing_errors']:>10,} errors                         │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────── FINAL RESULTS ───────────────────────────────┐
│ Passed All Filters     : {results['passed_all_filters']:>10,} datasets                     │
│ Overall Pass Rate      : {results['overall_pass_rate']:>9.2f}% of all samples             │
│ Output File            : {results['output_file']:<40} │
└────────────────────────────────────────────────────────────────────┘
"""
        
        logger.info(final_report)
        
        # Add success/warning message
        if results['passed_all_filters'] > 0:
            success_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              🎉 SUCCESS 🎉                                   ║
║                                                                              ║
║  Found {results['passed_all_filters']:>6,} high-quality human single-cell datasets suitable      ║
║  for sc-eQTL analysis!                                                       ║
║                                                                              ║
║  Next steps:                                                                 ║
║  1. Review the output CSV file for detailed results                          ║
║  2. Check the statistics JSON file for processing metadata                   ║
║  3. Consider running deeper analysis on filtered datasets                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.info(success_msg)
        else:
            warning_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ⚠️  WARNING ⚠️                                   ║
║                                                                              ║
║  No samples passed all filters!                                              ║
║                                                                              ║
║  Suggestions:                                                                ║
║  1. Review filter criteria - may be too restrictive                          ║
║  2. Check data quality in the source table                                   ║
║  3. Enable AI assistance if not already enabled                              ║
║  4. Consider processing a larger sample set                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.info(warning_msg)

def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Human Single-Cell Analysis for sc-eQTL (v2.0 Parallel)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run with 1000 samples
  python comprehensive_human_sc_analysis.py --test-run

  # Process 10000 samples with larger batches
  python comprehensive_human_sc_analysis.py --quick-scan 10000 --batch-size 2000

  # Full scan of entire database
  python comprehensive_human_sc_analysis.py --fullscan

  # Custom output with AI disabled
  python comprehensive_human_sc_analysis.py --output results.csv --no-ai
  
  # High-performance mode with more AI workers
  python comprehensive_human_sc_analysis.py --quick-scan 50000 --ai-workers 15
        """
    )
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Maximum samples to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size for processing (default: 5000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file (default: auto-generated with timestamp)')
    parser.add_argument('--no-ai', action='store_true',
                       help='Disable AI-assisted analysis')
    parser.add_argument('--test-run', action='store_true',
                       help='Run with limited samples for testing (1000 samples)')
    parser.add_argument('--fullscan', action='store_true',
                       help='Perform full database scan (all records)')
    parser.add_argument('--quick-scan', type=int, default=None,
                       help='Quick scan with specified number of samples (e.g., 10000)')
    parser.add_argument('--ai-workers', type=int, default=15,
                       help='Number of parallel AI workers (default: 15, max: 15)')
    
    args = parser.parse_args()
    
    # Validate AI workers
    args.ai_workers = min(max(1, args.ai_workers), 15)
    
    # Configure sample limits based on arguments
    if args.test_run:
        args.max_samples = 1000
        args.batch_size = 200
        print("\n" + "="*80)
        print("TEST RUN MODE: Limited to 1000 samples")
        print("="*80 + "\n")
    elif args.quick_scan:
        args.max_samples = args.quick_scan
        args.batch_size = min(2000, args.quick_scan // 5)  # Optimize batch size
        print("\n" + "="*80)
        print(f"QUICK SCAN MODE: Processing {args.quick_scan:,} samples")
        print("="*80 + "\n")
    elif args.fullscan:
        args.max_samples = None  # No limit
        args.batch_size = 10000  # Large batches for full scan
        print("\n" + "="*80)
        print("FULL DATABASE SCAN MODE: Processing ALL samples")
        print("="*80 + "\n")
    
    # Initialize analysis system with AI workers configuration
    analysis = ComprehensiveHumanScAnalysis(max_ai_workers=args.ai_workers)
    
    # Run comprehensive analysis
    result = analysis.run_comprehensive_analysis(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_file=args.output,
        enable_ai=not args.no_ai
    )
    
    # Exit with appropriate code
    if result.get("status") == "completed" and result.get("results", {}).get("passed_all_filters", 0) > 0:
        sys.exit(0)  # Success with results
    elif result.get("status") == "completed":
        sys.exit(1)  # Completed but no results
    else:
        sys.exit(2)  # Failed

if __name__ == "__main__":
    main() 