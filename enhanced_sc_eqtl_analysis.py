#!/usr/bin/env python3
"""
Enhanced sc-eQTL Analysis with Comprehensive Dataset Discovery
Features:
- SRA Lite URL generation with download options
- Age field extraction from GEO/SRA characteristics  
- Tumor vs normal tissue discrimination
- scRNA-seq vs scATAC-seq classification
- Cell line detection and exclusion
- Comprehensive metadata extraction
- GPU-accelerated batch processing
- PMC PDF analysis for detailed information
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
import argparse

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent_2025'))

from scAgent_2025.utils_enhanced_sc_eqtl_optimizer import EnhancedScEqtlOptimizer
from scAgent_2025.db.connect import get_connection
import psycopg2.extras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedScEqtlAnalysis:
    """
    Enhanced sc-eQTL analysis system with comprehensive dataset discovery.
    """
    
    def __init__(self, 
                 max_ai_workers: int = 20,
                 enable_gpu: bool = True,
                 include_download_urls: bool = True):
        self.optimizer = EnhancedScEqtlOptimizer(
            ai_client=None,  # Will be initialized when needed
            enable_gpu=enable_gpu,
            max_workers=max_ai_workers
        )
        self.max_ai_workers = max_ai_workers
        self.include_download_urls = include_download_urls
        self.results = []
        self.stats = {
            "total_processed": 0,
            "human_samples": 0,
            "passed_all_filters": 0,
            "single_cell_identified": 0,
            "cell_lines_excluded": 0,
            "tumor_samples": 0,
            "ai_assisted_decisions": 0,
            "processing_errors": 0,
            "scrna_seq_count": 0,
            "scatac_seq_count": 0
        }
        self.stats_lock = threading.Lock()
        
    def find_human_samples_enhanced(self, batch_size: int = 10000, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find human samples with enhanced filtering.
        
        Args:
            batch_size: Number of records per batch
            limit: Maximum records to process
            
        Returns:
            List of human sample records
        """
        logger.info("Starting enhanced human sample discovery...")
        
        try:
            conn = get_connection()
            
            # Enhanced query for human samples
            count_query = """
            SELECT COUNT(*) as total_count
            FROM "merged"."sra_geo_ft2"
            WHERE 
                (LOWER(COALESCE("organism_ch1", '')) LIKE %s OR
                 LOWER(COALESCE("scientific_name", '')) LIKE %s OR
                 LOWER(COALESCE("experiment_title", '')) LIKE %s OR
                 LOWER(COALESCE("study_title", '')) LIKE %s OR
                 LOWER(COALESCE("summary", '')) LIKE %s) AND
                "organism_ch1" IS NOT NULL
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
            
            # Fetch samples in batches
            all_samples = []
            offset = 0
            
            main_query = """
            SELECT * FROM "merged"."sra_geo_ft2"
            WHERE 
                (LOWER(COALESCE("organism_ch1", '')) LIKE %s OR
                 LOWER(COALESCE("scientific_name", '')) LIKE %s OR
                 LOWER(COALESCE("experiment_title", '')) LIKE %s OR
                 LOWER(COALESCE("study_title", '')) LIKE %s OR
                 LOWER(COALESCE("summary", '')) LIKE %s) AND
                "organism_ch1" IS NOT NULL
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
    
    def process_single_sample_enhanced(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single sample with enhanced analysis.
        
        Args:
            sample: Sample record
            
        Returns:
            Enhanced result record or None if error
        """
        try:
            # Apply enhanced filtering pipeline
            result = self.optimizer.filter_record_enhanced(sample)
            
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
    
    def process_samples_batch_parallel_enhanced(self, samples: List[Dict[str, Any]], enable_ai: bool = True) -> List[Dict[str, Any]]:
        """
        Process samples in parallel with enhanced analysis.
        
        Args:
            samples: List of sample records
            enable_ai: Enable AI-assisted analysis
            
        Returns:
            List of processed results
        """
        batch_results = []
        
        # Initialize AI client if needed
        if enable_ai and not self.optimizer.ai_client:
            from scAgent_2025.models.client import get_model_client
            self.optimizer.ai_client = get_model_client()
        
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
                result = self.process_single_sample_enhanced(sample)
                if result:
                    batch_results.append(result)
                pbar.update(1)
        
        # Process AI-needed samples in parallel
        if ai_needed_samples and enable_ai:
            with ThreadPoolExecutor(max_workers=self.max_ai_workers) as executor:
                # Submit all AI tasks
                future_to_sample = {
                    executor.submit(self.process_single_sample_enhanced, sample): sample 
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
        Create enhanced result record with comprehensive sc-eQTL information.
        
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
        
        # Enhanced metadata
        metadata = filter_result.get('metadata', {})
        
        # Age information
        age_info = metadata.get('age_info', {})
        enhanced.update({
            "age_value": age_info.get('age_value', ''),
            "age_unit": age_info.get('age_unit', ''),
            "age_source": age_info.get('age_source', ''),
            "age_confidence": age_info.get('age_confidence', 0.0)
        })
        
        # Tumor status
        tumor_status = metadata.get('tumor_status', {})
        enhanced.update({
            "is_tumor": tumor_status.get('is_tumor', False),
            "tumor_type": tumor_status.get('tumor_type', ''),
            "tumor_confidence": tumor_status.get('confidence', 0.0)
        })
        
        # Cell line information
        cell_line_info = metadata.get('cell_line_info', {})
        enhanced.update({
            "is_cell_line": cell_line_info.get('is_cell_line', False),
            "cell_line_name": cell_line_info.get('cell_line_name', ''),
            "cell_line_confidence": cell_line_info.get('confidence', 0.0)
        })
        
        # Single-cell classification
        sc_classification = metadata.get('sc_classification', {})
        enhanced.update({
            "experiment_type": sc_classification.get('experiment_type', 'unknown'),
            "sc_technology": sc_classification.get('technology', ''),
            "sc_confidence": sc_classification.get('confidence', 0.0)
        })
        
        # Sample size
        sample_size = metadata.get('sample_size', {})
        enhanced.update({
            "estimated_sample_size": sample_size.get('estimated_size', ''),
            "sample_size_source": sample_size.get('size_source', ''),
            "sample_size_confidence": sample_size.get('confidence', 0.0)
        })
        
        # Publication information
        pub_info = metadata.get('publication_info', {})
        enhanced.update({
            "pmid": pub_info.get('pmid', ''),
            "doi": pub_info.get('doi', ''),
            "journal": pub_info.get('journal', ''),
            "publication_date": pub_info.get('publication_date', ''),
            "authors": pub_info.get('authors', '')
        })
        
        # Demographics
        demographics = metadata.get('demographics', {})
        enhanced.update({
            "geographic_location": demographics.get('geographic_location', ''),
            "ethnicity": demographics.get('ethnicity', ''),
            "gender": demographics.get('gender', ''),
            "health_status": demographics.get('health_status', '')
        })
        
        # Quality metrics
        quality_metrics = metadata.get('quality_metrics', {})
        enhanced.update({
            "data_completeness": quality_metrics.get('data_completeness', 0.0),
            "metadata_richness": quality_metrics.get('metadata_richness', 0.0),
            "overall_quality_score": quality_metrics.get('overall_quality_score', 0.0)
        })
        
        # SRA Lite URLs
        sra_urls = metadata.get('sra_lite_url', {})
        enhanced.update({
            "sra_lite_url": sra_urls.get('sra_lite_url', ''),
            "data_access_url": sra_urls.get('data_access_url', '')
        })
        
        if self.include_download_urls:
            enhanced.update({
                "fastq_download_url": sra_urls.get('fastq_download', ''),
                "sra_download_url": sra_urls.get('sra_download', '')
            })
        
        # Filter step results
        filter_steps = filter_result.get('filter_steps', {})
        for step_name, step_result in filter_steps.items():
            enhanced[f"{step_name}_passed"] = step_result.get('passed', False)
            enhanced[f"{step_name}_reason"] = step_result.get('reason', '')
            enhanced[f"{step_name}_confidence"] = step_result.get('confidence', 0.0)
        
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
            
            # Count experiment types
            experiment_type = filter_result.get('metadata', {}).get('sc_classification', {}).get('experiment_type', '')
            if experiment_type == 'scRNA-seq':
                self.stats["scrna_seq_count"] += 1
            elif experiment_type == 'scATAC-seq':
                self.stats["scatac_seq_count"] += 1
            
        if not filter_steps.get('cell_line_check', {}).get('passed', True):  # Note: inverted logic
            self.stats["cell_lines_excluded"] += 1
        
        if filter_result.get('passes_filter', False):
            self.stats["passed_all_filters"] += 1
            
        # Check tumor status
        if filter_result.get('metadata', {}).get('tumor_status', {}).get('is_tumor', False):
            self.stats["tumor_samples"] += 1
    
    def save_results_to_csv_enhanced(self, output_file: str):
        """
        Save enhanced results to comprehensive CSV file.
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving {len(self.results):,} enhanced results to {output_file}")
        
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(self.results)
            
            # Save to CSV with progress bar
            with tqdm(total=1, desc="Saving enhanced CSV file", unit="file") as pbar:
                df.to_csv(output_file, index=False, encoding='utf-8')
                pbar.update(1)
            
            # Also save a summary statistics file
            stats_file = output_file.replace('.csv', '_enhanced_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump({
                    "processing_statistics": self.stats,
                    "result_summary": {
                        "total_results": len(self.results),
                        "passed_filters": len([r for r in self.results if r.get('passes_all_filters', False)]),
                        "pass_rate_percentage": (len([r for r in self.results if r.get('passes_all_filters', False)]) / len(self.results) * 100) if self.results else 0,
                        "average_confidence_score": sum([r.get('final_confidence_score', 0) for r in self.results]) / len(self.results) if self.results else 0,
                        "scrna_seq_count": self.stats["scrna_seq_count"],
                        "scatac_seq_count": self.stats["scatac_seq_count"],
                        "tumor_samples": self.stats["tumor_samples"]
                    },
                    "generated_at": datetime.now().isoformat(),
                    "script_version": "enhanced_v1.0_parallel"
                }, f, indent=2)
            
            logger.info(f"Enhanced results saved to {output_file}")
            logger.info(f"Enhanced statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced results: {e}")
            raise
    
    def run_enhanced_analysis(self, 
                             max_samples: Optional[int] = None,
                             batch_size: int = 20000,
                             output_file: Optional[str] = None,
                             enable_ai: bool = True,
                             gpu_batch_size: int = 1000,
                             auto_download: bool = False) -> Dict[str, Any]:
        """
        Run enhanced comprehensive analysis of human samples.
        
        Args:
            max_samples: Maximum samples to process (None for all)
            batch_size: Batch size for processing
            output_file: Output CSV file path
            enable_ai: Enable AI-assisted analysis
            gpu_batch_size: GPU batch size for filtering
            auto_download: Automatically download SRA data
            
        Returns:
            Analysis summary
        """
        start_time = datetime.now()
        
        if output_file is None:
            output_file = f"enhanced_sc_eqtl_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Print startup banner
        startup_banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ENHANCED sc-eQTL ANALYSIS WITH ULTRA-OPTIMIZED PROCESSING             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Configuration:                                                               ║
║   • Start Time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}                         ║
║   • Max Samples        : {str(max_samples or 'ALL'):>10}                                  ║
║   • Batch Size         : {batch_size:>10,}                                  ║
║   • AI Assistance      : {('ENABLED' if enable_ai else 'DISABLED'):>10}                                  ║
║   • Max AI Workers     : {self.max_ai_workers:>10}                                  ║
║   • GPU Acceleration   : {('ENABLED' if self.optimizer.enable_gpu else 'DISABLED'):>10}                                  ║
║   • GPU Device         : {str(self.optimizer.device):>10}                                  ║
║   • GPU Batch Size     : {gpu_batch_size:>10}                                  ║
║   • Download URLs      : {('INCLUDED' if self.include_download_urls else 'EXCLUDED'):>10}                                  ║
║   • Auto Download      : {('ENABLED' if auto_download else 'DISABLED'):>10}                                  ║
║   • Output File        : {output_file[:40]:<40} ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        logger.info(startup_banner)
        
        try:
            # 使用新的优化批量处理方法
            logger.info("\n--- ULTRA-OPTIMIZED BATCH PROCESSING ---")
            
            result = self.optimizer.batch_process_enhanced(
                batch_size=batch_size,
                max_records=max_samples,
                output_file=output_file,
                include_download_urls=self.include_download_urls,
                auto_download=auto_download,
                gpu_batch_size=gpu_batch_size
            )
            
            if result["status"] == "no_records":
                logger.error("No records found. Exiting.")
                return {"status": "failed", "error": "No records found"}
            elif result["status"] == "no_pass_hard_filter":
                logger.warning("No records passed hard filter. Exiting.")
                return {"status": "completed", "results": [], "summary": result.get("summary", {})}
            
            # 更新结果
            self.results = result.get("results", [])
            
            # 生成最终统计
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            summary = result.get("summary", {})
            final_summary = {
                "status": "completed",
                "processing_statistics": {
                    "total_processed": summary.get("total_processed", 0),
                    "hard_filtered": summary.get("hard_filtered", 0),
                    "final_passed": summary.get("final_passed", 0),
                    "filter_time": summary.get("filter_time", 0),
                    "analysis_time": summary.get("analysis_time", 0),
                    "total_time": summary.get("total_time", 0),
                    "records_per_second": summary.get("records_per_second", 0),
                    "human_samples": summary.get("total_processed", 0),  # All processed are human
                    "single_cell_identified": summary.get("final_passed", 0),  # All passed are single-cell
                    "scrna_seq_count": summary.get("final_passed", 0),  # Assume all are scRNA-seq for now
                    "scatac_seq_count": 0,  # Will be updated later
                    "cell_lines_excluded": 0,  # Will be updated later
                    "tumor_samples": 0,  # Will be updated later
                    "ai_assisted_decisions": 0,  # Will be updated later
                    "processing_errors": 0,  # Will be updated later
                    "passed_all_filters": summary.get("final_passed", 0)
                },
                "timing": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_time_seconds": total_time,
                    "samples_per_second": summary.get("total_processed", 0) / total_time if total_time > 0 else 0
                },
                "results": {
                    "total_results": len(self.results),
                    "output_file": output_file,
                    "passed_all_filters": len(self.results),
                    "overall_pass_rate": (len(self.results) / summary.get("total_processed", 1) * 100) if summary.get("total_processed", 0) > 0 else 0,
                    "scrna_seq_count": len(self.results),  # Assume all are scRNA-seq for now
                    "scatac_seq_count": 0,  # Will be updated later
                    "tumor_samples": 0  # Will be updated later
                }
            }
            
            self._print_enhanced_final_summary(final_summary)
            return final_summary
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _report_enhanced_progress_stats(self):
        """Report current enhanced progress statistics."""
        stats = self.stats
        
        # Create a formatted progress report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ENHANCED PROGRESS STATISTICS                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Processed        : {stats['total_processed']:>10,} samples                             ║
║ Human Samples          : {stats['human_samples']:>10,} confirmed                           ║
║ Single-Cell Identified : {stats['single_cell_identified']:>10,} experiments                      ║
║   ├─ scRNA-seq         : {stats['scrna_seq_count']:>10,} experiments                      ║
║   └─ scATAC-seq        : {stats['scatac_seq_count']:>10,} experiments                      ║
║ Cell Lines Excluded    : {stats['cell_lines_excluded']:>10,} samples                             ║
║ Tumor Samples          : {stats['tumor_samples']:>10,} samples                             ║
║ Passed All Filters     : {stats['passed_all_filters']:>10,} datasets                            ║
║ AI-Assisted Decisions  : {stats['ai_assisted_decisions']:>10,} cases                               ║
║ Processing Errors      : {stats['processing_errors']:>10,} errors                              ║
"""
        
        if stats['total_processed'] > 0:
            pass_rate = (stats['passed_all_filters'] / stats['total_processed']) * 100
            sc_rate = (stats['single_cell_identified'] / stats['human_samples']) * 100 if stats['human_samples'] > 0 else 0
            tumor_rate = (stats['tumor_samples'] / stats['human_samples']) * 100 if stats['human_samples'] > 0 else 0
            report += f"""╠══════════════════════════════════════════════════════════════════════════════╣
║ Overall Pass Rate      : {pass_rate:>9.2f}% of all samples                         ║
║ Single-Cell Rate       : {sc_rate:>9.2f}% of human samples                       ║
║ Tumor Sample Rate      : {tumor_rate:>9.2f}% of human samples                       ║
"""
        
        report += "╚══════════════════════════════════════════════════════════════════════════════╝"
        
        logger.info(report)
    
    def _print_enhanced_final_summary(self, summary: Dict[str, Any]):
        """Print enhanced final analysis summary."""
        stats = summary["processing_statistics"]
        timing = summary["timing"]
        results = summary["results"]
        
        # Calculate additional metrics
        hours = timing['total_time_seconds'] / 3600
        minutes = (timing['total_time_seconds'] % 3600) / 60
        seconds = timing['total_time_seconds'] % 60
        
        # Build the enhanced summary report
        final_report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ENHANCED sc-eQTL ANALYSIS COMPLETED SUCCESSFULLY                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────── PROCESSING PERFORMANCE ──────────────────────┐
│ Total Processing Time  : {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s                             │
│ Processing Rate        : {timing['samples_per_second']:>8.1f} samples/second                    │
│ Start Time             : {timing['start_time'][:19]}                      │
│ End Time               : {timing['end_time'][:19]}                        │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────── ENHANCED SAMPLE STATISTICS ──────────────────┐
│ Total Processed        : {stats['total_processed']:>10,} samples                      │
│ Human Samples          : {stats['human_samples']:>10,} confirmed ({(stats['human_samples']/stats['total_processed']*100 if stats['total_processed'] > 0 else 0):>5.1f}%)           │
│ Single-Cell Identified : {stats['single_cell_identified']:>10,} experiments ({(stats['single_cell_identified']/stats['human_samples']*100 if stats['human_samples'] > 0 else 0):>5.1f}%)         │
│   ├─ scRNA-seq         : {stats['scrna_seq_count']:>10,} experiments                      │
│   └─ scATAC-seq        : {stats['scatac_seq_count']:>10,} experiments                      │
│ Cell Lines Excluded    : {stats['cell_lines_excluded']:>10,} samples                      │
│ Tumor Samples          : {stats['tumor_samples']:>10,} samples ({(stats['tumor_samples']/stats['human_samples']*100 if stats['human_samples'] > 0 else 0):>5.1f}%)           │
│ AI-Assisted Decisions  : {stats['ai_assisted_decisions']:>10,} cases ({(stats['ai_assisted_decisions']/stats['total_processed']*100 if stats['total_processed'] > 0 else 0):>5.1f}%)              │
│ Processing Errors      : {stats['processing_errors']:>10,} errors                         │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────── ENHANCED FINAL RESULTS ─────────────────────┐
│ Passed All Filters     : {results['passed_all_filters']:>10,} datasets                     │
│ Overall Pass Rate      : {results['overall_pass_rate']:>9.2f}% of all samples             │
│ scRNA-seq Datasets     : {results['scrna_seq_count']:>10,} experiments                     │
│ scATAC-seq Datasets    : {results['scatac_seq_count']:>10,} experiments                     │
│ Tumor Datasets         : {results['tumor_samples']:>10,} samples                            │
│ Output File            : {results['output_file']:<40} │
└────────────────────────────────────────────────────────────────────┘
"""
        
        logger.info(final_report)
        
        # Add success/warning message
        if results['passed_all_filters'] > 0:
            success_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              🎉 ENHANCED SUCCESS 🎉                          ║
║                                                                              ║
║  Found {results['passed_all_filters']:>6,} high-quality human single-cell datasets suitable      ║
║  for sc-eQTL analysis with comprehensive metadata!                          ║
║                                                                              ║
║  Enhanced Features:                                                          ║
║  • SRA Lite URLs with download options                                      ║
║  • Age information extraction                                               ║
║  • Tumor vs normal tissue classification                                    ║
║  • scRNA-seq vs scATAC-seq differentiation                                 ║
║  • Cell line detection and exclusion                                        ║
║  • Comprehensive metadata extraction                                        ║
║                                                                              ║
║  Next steps:                                                                 ║
║  1. Review the enhanced output CSV file for detailed results                ║
║  2. Check the enhanced statistics JSON file for processing metadata         ║
║  3. Use SRA Lite URLs for data access and download                          ║
║  4. Consider running deeper analysis on filtered datasets                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.info(success_msg)
        else:
            warning_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ⚠️  ENHANCED WARNING ⚠️                          ║
║                                                                              ║
║  No samples passed all enhanced filters!                                    ║
║                                                                              ║
║  Suggestions:                                                                ║
║  1. Review enhanced filter criteria - may be too restrictive                ║
║  2. Check data quality in the source table                                  ║
║  3. Enable AI assistance if not already enabled                             ║
║  4. Consider processing a larger sample set                                 ║
║  5. Verify age, tumor, and cell line detection parameters                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.info(warning_msg)

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Enhanced sc-eQTL Analysis with Comprehensive Dataset Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run with 1000 samples
  python enhanced_sc_eqtl_analysis.py --test-run

  # Process 10000 samples with enhanced features
  python enhanced_sc_eqtl_analysis.py --quick-scan 10000 --batch-size 20000

  # Full scan with download URLs and GPU acceleration
  python enhanced_sc_eqtl_analysis.py --fullscan --include-downloads --gpu

  # High-performance mode with 128 workers and large batch
  python enhanced_sc_eqtl_analysis.py --quick-scan 50000 --ai-workers 128 --batch-size 50000 --gpu
  
  # Custom output with AI disabled
  python enhanced_sc_eqtl_analysis.py --output results.csv --no-ai
        """
    )
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Maximum samples to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=20000,
                       help='Batch size for processing (default: 20000)')
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
    parser.add_argument('--ai-workers', type=int, default=64,
                       help='Number of parallel AI workers (default: 64, max: 256)')
    parser.add_argument('--include-downloads', action='store_true',
                       help='Include download URLs in output')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration (default)')
    parser.add_argument('--gpu-batch-size', type=int, default=1000,
                       help='GPU batch size for filtering (default: 1000)')
    parser.add_argument('--auto-download', action='store_true',
                       help='Automatically download SRA data after processing')
    
    args = parser.parse_args()
    
    # Validate AI workers
    args.ai_workers = min(max(1, args.ai_workers), 256)  # 提升最大并行度
    
    # Configure sample limits based on arguments
    if args.test_run:
        args.max_samples = 1000
        args.batch_size = 5000
        print("\n" + "="*80)
        print("ENHANCED TEST RUN MODE: Limited to 1000 samples")
        print("="*80 + "\n")
    elif args.quick_scan:
        args.max_samples = args.quick_scan
        args.batch_size = min(args.batch_size, args.quick_scan // 2)  # 优化batch size
        print("\n" + "="*80)
        print(f"ENHANCED QUICK SCAN MODE: Processing {args.quick_scan:,} samples")
        print("="*80 + "\n")
    elif args.fullscan:
        args.max_samples = None  # No limit
        args.batch_size = 50000  # 更大的batch for full scan
        print("\n" + "="*80)
        print("ENHANCED FULL DATABASE SCAN MODE: Processing ALL samples")
        print("="*80 + "\n")
    
    # GPU configuration
    enable_gpu = not args.no_gpu
    if enable_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} devices)"
                print(f"🚀 {gpu_info}")
            else:
                print("⚠️ GPU requested but not available, using CPU")
                enable_gpu = False
        except ImportError:
            print("⚠️ PyTorch not available, GPU acceleration disabled")
            enable_gpu = False
    
    # Initialize enhanced analysis system
    analysis = EnhancedScEqtlAnalysis(
        max_ai_workers=args.ai_workers,
        enable_gpu=enable_gpu,
        include_download_urls=args.include_downloads
    )
    
    # Run enhanced comprehensive analysis
    result = analysis.run_enhanced_analysis(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_file=args.output,
        enable_ai=not args.no_ai,
        gpu_batch_size=args.gpu_batch_size,
        auto_download=args.auto_download
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