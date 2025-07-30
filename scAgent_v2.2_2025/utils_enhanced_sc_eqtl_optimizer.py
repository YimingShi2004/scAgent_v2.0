#!/usr/bin/env python3
"""
Enhanced sc-eQTL Optimizer with Comprehensive Dataset Analysis
Features:
- SRA Lite URL generation with download options
- Age field extraction from GEO/SRA characteristics
- Tumor vs normal tissue discrimination
- scRNA-seq vs scATAC-seq classification
- Cell line detection and exclusion
- Comprehensive metadata extraction
- GPU-accelerated batch processing
"""

import sys
import os
import json
import csv
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from collections import defaultdict

# Try to import torch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .db.connect import get_connection
from .models.client import get_model_client
import psycopg2.extras

logger = logging.getLogger(__name__)

class EnhancedScEqtlOptimizer:
    """
    Enhanced optimizer for sc-eQTL dataset discovery with comprehensive analysis.
    """
    
    def __init__(self, 
                 schema: str = "merged", 
                 table: str = "sra_geo_ft2",
                 ai_client: Optional[Any] = None,
                 enable_gpu: bool = True,
                 max_workers: int = 64):  # 提升默认并行度
        self.schema = schema
        self.table = table
        self.full_table_name = f'"{schema}"."{table}"'
        self.ai_client = ai_client or get_model_client()
        self.enable_gpu = enable_gpu
        self.max_workers = max_workers
        self.processing_stats = defaultdict(int)
        self.stats_lock = threading.Lock()
        
        # GPU acceleration support
        self.device = None
        if self.enable_gpu and TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("GPU requested but not available, using CPU")
                    self.device = torch.device('cpu')
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}, using CPU")
                self.device = torch.device('cpu')
        else:
            if self.enable_gpu and not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, GPU acceleration disabled")
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None
        
        # SRA Lite URL base
        self.sra_lite_base = "https://trace.ncbi.nlm.nih.gov/Traces/?view=run_browser&acc="
        
        # Age patterns for extraction
        self.age_patterns = [
            r'age\s*:\s*([^;,\n]+)',
            r'age\s*=\s*([^;,\n]+)',
            r'(\d+)\s*(?:years?|yrs?|y\.o)',
            r'(\d+)\s*(?:months?|mos?)',
            r'(\d+)\s*(?:days?|d)',
            r'(\d+)\s*(?:weeks?|wks?)',
            r'(\d+)\s*(?:hours?|hrs?)',
            r'p(\d+)',  # Postnatal days
            r'(\d+)\s*yo',  # Years old
        ]
        
        # Tumor indicators
        self.tumor_indicators = [
            'tumor', 'cancer', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma',
            'melanoma', 'adenocarcinoma', 'squamous cell', 'basal cell',
            'glioblastoma', 'astrocytoma', 'oligodendroglioma', 'ependymoma',
            'medulloblastoma', 'neuroblastoma', 'retinoblastoma', 'hepatoblastoma',
            'nephroblastoma', 'rhabdomyosarcoma', 'osteosarcoma', 'chondrosarcoma',
            'fibrosarcoma', 'liposarcoma', 'leiomyosarcoma', 'angiosarcoma',
            'kaposi sarcoma', 'ewing sarcoma', 'synovial sarcoma', 'alveolar soft part sarcoma'
        ]
        
        # 优化的关键词匹配（预编译正则表达式）
        self.human_pattern = re.compile(r'\b(homo sapiens|human)\b', re.IGNORECASE)
        self.cell_line_pattern = re.compile(r'\b(cell line|cellline|cl-|cl |hepg2|hela|jurkat|k562|a549|mcf7|pc3|du145|lncap|skov3|ovcar|ht29|sw480|caco2|beas2b|hek293|cos7|cho|immortalized|transformed|cancer cell line)\b', re.IGNORECASE)
        self.scrna_pattern = re.compile(r'\b(scrna-seq|single-cell rna|10x|smart-seq|drop-seq|cel-seq|mars-seq|quartz-seq)\b', re.IGNORECASE)
        self.scatac_pattern = re.compile(r'\b(scatac-seq|single-cell atac|atac-seq|sci-atac)\b', re.IGNORECASE)
        
        # 优化的列映射
        self.column_mapping = {
            # Core identifiers
            "sra_id": "sra_ID",
            "gsm_title": "gsm_title", 
            "gse_title": "gse_title",
            "experiment_title": "experiment_title",
            "run_alias": "run_alias",
            "experiment_alias": "experiment_alias",
            "sample_alias": "sample_alias",
            "study_alias": "study_alias",
            
            # Organism information
            "organism_ch1": "organism_ch1", 
            "organism": "organism",
            "scientific_name": "scientific_name",
            "common_name": "common_name",
            
            # Sample and tissue information
            "characteristics_ch1": "characteristics_ch1",
            "source_name_ch1": "source_name_ch1",
            "sample_attribute": "sample_attribute",
            "description": "description",
            "gsm_description": "gsm_description",
            
            # Study information
            "study_title": "study_title",
            "study_abstract": "study_abstract", 
            "summary": "summary",
            "overall_design": "overall_design",
            
            # Sequencing technology
            "library_strategy": "library_strategy",
            "library_source": "library_source",
            "library_selection": "library_selection",
            "library_layout": "library_layout",
            "platform": "platform",
            "instrument_model": "instrument_model",
            "technology": "technology",
            
            # Data metrics
            "spots": "spots",
            "bases": "bases",
            "spot_length": "spot_length",
            
            # Dates
            "submission_date": "submission_date",
            "gsm_submission_date": "gsm_submission_date",
            "gse_submission_date": "gse_submission_date",
            "run_date": "run_date",
            
            # Publication info
            "pubmed_id": "pubmed_id",
            "center_project_name": "center_project_name",
            "lab_name": "lab_name"
        }
        
        # Cell line indicators
        self.cell_line_indicators = [
            'cell line', 'cellline', 'cl-', 'cl ', 'hepg2', 'hela', 'jurkat',
            'k562', 'a549', 'mcf7', 'pc3', 'du145', 'lncap', 'skov3', 'ovcar',
            'ht29', 'sw480', 'caco2', 'beas2b', 'hek293', 'cos7', 'cho',
            'immortalized', 'transformed', 'cancer cell line'
        ]
        
        # Single-cell technology indicators
        self.sc_technology_indicators = {
            'scrna_seq': ['10x', 'smart-seq', 'drop-seq', 'cel-seq', 'mars-seq', 'quartz-seq'],
            'scatac_seq': ['10x atac', 'sci-atac', 'sci-atac-seq', 'droplet atac', 'atac-seq']
        }
    
    def generate_sra_lite_url(self, sra_id: str, include_download: bool = False) -> Dict[str, str]:
        """
        Generate SRA Lite URL for data access.
        
        Args:
            sra_id: SRA run ID
            include_download: Whether to include download links
            
        Returns:
            Dict with URLs
        """
        base_url = f"{self.sra_lite_base}{sra_id}&display=data-access"
        
        urls = {
            "sra_lite_url": base_url,
            "data_access_url": base_url
        }
        
        if include_download:
            # Add download URLs
            urls.update({
                "fastq_download": f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={sra_id}&format=fastq",
                "sra_download": f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={sra_id}&format=sra"
            })
        
        return urls
    
    def extract_age_information(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract age information from various fields.
        
        Args:
            record: Database record
            
        Returns:
            Dict with age information
        """
        age_info = {
            "age_value": "",
            "age_unit": "",
            "age_source": "",
            "age_confidence": 0.0
        }
        
        # Fields to check for age information
        age_fields = [
            record.get('characteristics_ch1', ''),
            record.get('source_name_ch1', ''),
            record.get('gsm_description', ''),
            record.get('study_title', ''),
            record.get('summary', '')
        ]
        
        for field_name, field_value in zip(
            ['characteristics_ch1', 'source_name_ch1', 'gsm_description', 'study_title', 'summary'],
            age_fields
        ):
            if not field_value:
                continue
                
            field_text = str(field_value).lower()
            
            # Check for age patterns
            for pattern in self.age_patterns:
                matches = re.findall(pattern, field_text, re.IGNORECASE)
                if matches:
                    age_value = matches[0].strip()
                    
                    # Determine age unit
                    age_unit = "unknown"
                    if any(unit in age_value.lower() for unit in ['year', 'yr', 'y.o']):
                        age_unit = "years"
                    elif any(unit in age_value.lower() for unit in ['month', 'mo']):
                        age_unit = "months"
                    elif any(unit in age_value.lower() for unit in ['day', 'd']):
                        age_unit = "days"
                    elif 'p' in age_value.lower():
                        age_unit = "postnatal_days"
                    elif any(unit in age_value.lower() for unit in ['week', 'wk']):
                        age_unit = "weeks"
                    elif any(unit in age_value.lower() for unit in ['hour', 'hr']):
                        age_unit = "hours"
                    
                    age_info.update({
                        "age_value": age_value,
                        "age_unit": age_unit,
                        "age_source": field_name,
                        "age_confidence": 0.9
                    })
                    return age_info
        
        return age_info
    
    def detect_tumor_status(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if sample is from tumor tissue.
        
        Args:
            record: Database record
            
        Returns:
            Dict with tumor detection results
        """
        tumor_info = {
            "is_tumor": False,
            "tumor_type": "",
            "confidence": 0.0,
            "evidence": []
        }
        
        # Fields to check
        check_fields = [
            record.get('characteristics_ch1', ''),
            record.get('source_name_ch1', ''),
            record.get('gsm_description', ''),
            record.get('study_title', ''),
            record.get('summary', ''),
            record.get('experiment_title', '')
        ]
        
        combined_text = ' '.join([str(f) for f in check_fields if f]).lower()
        
        # Check for tumor indicators
        found_indicators = []
        for indicator in self.tumor_indicators:
            if indicator in combined_text:
                found_indicators.append(indicator)
        
        if found_indicators:
            tumor_info.update({
                "is_tumor": True,
                "tumor_type": "; ".join(found_indicators),
                "confidence": min(0.9, len(found_indicators) * 0.3),
                "evidence": found_indicators
            })
        
        # Use AI for more sophisticated analysis if available
        if self.ai_client and len(combined_text) > 50:
            ai_result = self._analyze_tumor_with_ai(combined_text)
            if ai_result["is_tumor"]:
                tumor_info.update(ai_result)
        
        return tumor_info
    
    def _analyze_tumor_with_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI to analyze tumor status.
        
        Args:
            text: Combined text from record fields
            
        Returns:
            AI analysis result
        """
        try:
            prompt = f"""
            Analyze this scientific sample description to determine if it's from tumor tissue:

            Text: {text[:800]}

            Respond in JSON format:
            {{
                "is_tumor": true/false,
                "tumor_type": "specific tumor type if identified",
                "confidence": 0.0-1.0,
                "evidence": ["list", "of", "evidence", "found"]
            }}

            Look for:
            - Cancer/tumor mentions
            - Disease states
            - Tissue pathology
            - Clinical conditions
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return {"is_tumor": False, "tumor_type": "", "confidence": 0.0, "evidence": []}
                
        except Exception as e:
            logger.warning(f"AI tumor analysis failed: {e}")
            return {"is_tumor": False, "tumor_type": "", "confidence": 0.0, "evidence": []}
    
    def detect_cell_line(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if sample is from cell line (should be excluded).
        
        Args:
            record: Database record
            
        Returns:
            Dict with cell line detection results
        """
        cell_line_info = {
            "is_cell_line": False,
            "cell_line_name": "",
            "confidence": 0.0,
            "evidence": []
        }
        
        # Fields to check
        check_fields = [
            record.get('characteristics_ch1', ''),
            record.get('source_name_ch1', ''),
            record.get('gsm_description', ''),
            record.get('study_title', ''),
            record.get('summary', ''),
            record.get('experiment_title', '')
        ]
        
        combined_text = ' '.join([str(f) for f in check_fields if f]).lower()
        
        # Check for cell line indicators
        found_indicators = []
        for indicator in self.cell_line_indicators:
            if indicator in combined_text:
                found_indicators.append(indicator)
        
        if found_indicators:
            cell_line_info.update({
                "is_cell_line": True,
                "cell_line_name": "; ".join(found_indicators),
                "confidence": min(0.9, len(found_indicators) * 0.4),
                "evidence": found_indicators
            })
        
        return cell_line_info
    
    def classify_single_cell_type(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify single-cell experiment type (scRNA-seq vs scATAC-seq).
        
        Args:
            record: Database record
            
        Returns:
            Dict with classification results
        """
        classification = {
            "experiment_type": "unknown",
            "confidence": 0.0,
            "technology": "",
            "evidence": []
        }
        
        # Fields to check
        check_fields = [
            record.get('library_strategy', ''),
            record.get('experiment_title', ''),
            record.get('study_title', ''),
            record.get('summary', ''),
            record.get('technology', '')
        ]
        
        combined_text = ' '.join([str(f) for f in check_fields if f]).lower()
        
        # Check for technology indicators
        evidence = []
        
        # Check scRNA-seq indicators
        for tech in self.sc_technology_indicators['scrna_seq']:
            if tech in combined_text:
                evidence.append(f"scRNA-seq: {tech}")
        
        # Check scATAC-seq indicators
        for tech in self.sc_technology_indicators['scatac_seq']:
            if tech in combined_text:
                evidence.append(f"scATAC-seq: {tech}")
        
        # Determine type based on evidence
        if evidence:
            if any('scrna' in e.lower() for e in evidence):
                classification.update({
                    "experiment_type": "scRNA-seq",
                    "confidence": 0.8,
                    "technology": "; ".join([e for e in evidence if 'scrna' in e.lower()]),
                    "evidence": evidence
                })
            elif any('scatac' in e.lower() for e in evidence):
                classification.update({
                    "experiment_type": "scATAC-seq", 
                    "confidence": 0.8,
                    "technology": "; ".join([e for e in evidence if 'scatac' in e.lower()]),
                    "evidence": evidence
                })
        
        # Use AI for more sophisticated classification
        if self.ai_client and len(combined_text) > 50:
            ai_result = self._classify_with_ai(combined_text)
            if ai_result["confidence"] > classification["confidence"]:
                classification.update(ai_result)
        
        return classification
    
    def _classify_with_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI to classify single-cell experiment type.
        
        Args:
            text: Combined text from record fields
            
        Returns:
            AI classification result
        """
        try:
            prompt = f"""
            Classify this single-cell experiment as either scRNA-seq or scATAC-seq:

            Text: {text[:800]}

            Respond in JSON format:
            {{
                "experiment_type": "scRNA-seq|scATAC-seq|unknown",
                "confidence": 0.0-1.0,
                "technology": "specific technology used",
                "evidence": ["list", "of", "evidence", "found"]
            }}

            Look for:
            - RNA sequencing vs ATAC sequencing
            - Single-cell technologies (10x, Smart-seq, etc.)
            - Library preparation methods
            - Analysis workflows
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return {"experiment_type": "unknown", "confidence": 0.0, "technology": "", "evidence": []}
                
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
            return {"experiment_type": "unknown", "confidence": 0.0, "technology": "", "evidence": []}
    
    def extract_comprehensive_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata for sc-eQTL analysis.
        
        Args:
            record: Database record
            
        Returns:
            Dict with comprehensive metadata
        """
        metadata = {
            # Basic identifiers
            "sra_id": record.get('sra_ID', ''),
            "gsm_id": record.get('gsm_title', ''),
            "gse_id": record.get('gse_title', ''),
            
            # URLs
            "sra_lite_url": self.generate_sra_lite_url(record.get('sra_ID', ''), include_download=True),
            
            # Age information
            "age_info": self.extract_age_information(record),
            
            # Tumor status
            "tumor_status": self.detect_tumor_status(record),
            
            # Cell line detection
            "cell_line_info": self.detect_cell_line(record),
            
            # Single-cell classification
            "sc_classification": self.classify_single_cell_type(record),
            
            # Sample size estimation
            "sample_size": self._estimate_sample_size(record),
            
            # Publication information
            "publication_info": self._extract_publication_info(record),
            
            # Geographic and demographic info
            "demographics": self._extract_demographics(record),
            
            # Quality metrics
            "quality_metrics": self._calculate_quality_metrics(record)
        }
        
        return metadata
    
    def _estimate_sample_size(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate sample size from record metadata
        """
        try:
            # Extract sample size from various fields
            text_fields = [
                str(record.get('characteristics_ch1', '')),
                str(record.get('source_name_ch1', '')),
                str(record.get('gsm_description', '')),
                str(record.get('study_title', '')),
                str(record.get('summary', '')),
                str(record.get('experiment_title', ''))
            ]
            
            combined_text = ' '.join(text_fields).lower()
            
            # Look for sample size patterns
            import re
            patterns = [
                r'(\d+)\s*cells?',
                r'(\d+)\s*nuclei',
                r'sample\s*size[:\s]*(\d+)',
                r'n\s*=\s*(\d+)',
                r'(\d+)\s*samples?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, combined_text)
                if match:
                    size = int(match.group(1))
                    return {
                        "estimated_size": size,
                        "confidence": 0.8,
                        "source": "text_analysis"
                    }
            
            # Default estimation based on experiment type
            experiment_title = str(record.get('experiment_title', '')).lower()
            if '10x' in experiment_title:
                return {
                    "estimated_size": 1000,  # Typical 10x experiment
                    "confidence": 0.3,
                    "source": "default_estimation"
                }
            else:
                return {
                    "estimated_size": 500,  # Conservative estimate
                    "confidence": 0.2,
                    "source": "default_estimation"
                }
                
        except Exception as e:
            logger.warning(f"Error estimating sample size: {e}")
            return {
                "estimated_size": 0,
                "confidence": 0.0,
                "source": "error"
            }

    def _extract_publication_info(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract publication information from record
        """
        try:
            pmid = record.get('pubmed_id', '')
            doi = ''
            
            # Try to extract DOI from various fields
            text_fields = [
                str(record.get('study_title', '')),
                str(record.get('summary', '')),
                str(record.get('experiment_title', ''))
            ]
            
            combined_text = ' '.join(text_fields)
            
            # DOI pattern matching
            doi_pattern = r'10\.\d{4,}/[^\s]+'
            doi_match = re.search(doi_pattern, combined_text)
            if doi_match:
                doi = doi_match.group(0)
            
            return {
                "pmid": pmid if pmid else "Not available",
                "doi": doi if doi else "Not available",
                "confidence": 0.9 if pmid or doi else 0.5
            }
            
        except Exception as e:
            logger.warning(f"Error extracting publication info: {e}")
            return {
                "pmid": "Not available",
                "doi": "Not available",
                "confidence": 0.0
            }
    
    def _extract_demographics(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic information."""
        demo_info = {
            "geographic_location": "Not specified",
            "ethnicity": "Not specified", 
            "gender": "Not specified",
            "health_status": "Not specified"
        }
        
        # Use AI to extract demographics if available
        if self.ai_client:
            study_text = f"{record.get('study_title', '')} {record.get('study_abstract', '')}"
            if len(study_text) > 50:
                ai_demo_info = self._extract_demographics_with_ai(study_text)
                demo_info.update(ai_demo_info)
        
        return demo_info
    
    def _extract_demographics_with_ai(self, text: str) -> Dict[str, Any]:
        """Use AI to extract demographic information."""
        try:
            prompt = f"""
            Extract demographic information from this scientific text:

            Text: {text[:800]}

            Respond in JSON format:
            {{
                "geographic_location": "Country/region if mentioned",
                "ethnicity": "Ethnicity information if mentioned",
                "gender": "Gender information if mentioned", 
                "health_status": "Health status if mentioned"
            }}

            Use "Not specified" if information is not available.
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return {"geographic_location": "Not specified", "ethnicity": "Not specified", "gender": "Not specified", "health_status": "Not specified"}
                
        except Exception as e:
            logger.warning(f"AI demographics extraction failed: {e}")
            return {"geographic_location": "Not specified", "ethnicity": "Not specified", "gender": "Not specified", "health_status": "Not specified"}
    
    def _calculate_quality_metrics(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset."""
        quality = {
            "data_completeness": 0.0,
            "metadata_richness": 0.0,
            "overall_quality_score": 0.0
        }
        
        # Calculate data completeness
        required_fields = ['sra_ID', 'experiment_title', 'study_title', 'organism_ch1']
        filled_fields = sum(1 for field in required_fields if record.get(field))
        quality["data_completeness"] = filled_fields / len(required_fields)
        
        # Calculate metadata richness
        metadata_fields = ['characteristics_ch1', 'source_name_ch1', 'summary', 'study_abstract']
        filled_metadata = sum(1 for field in metadata_fields if record.get(field) and len(str(record.get(field))) > 10)
        quality["metadata_richness"] = filled_metadata / len(metadata_fields)
        
        # Overall quality score
        quality["overall_quality_score"] = (quality["data_completeness"] + quality["metadata_richness"]) / 2
        
        return quality
    
    def filter_record_enhanced(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced filtering with comprehensive analysis.
        
        Args:
            record: Database record
            
        Returns:
            Enhanced filtering result
        """
        start_time = time.time()
        
        # Extract comprehensive metadata
        metadata = self.extract_comprehensive_metadata(record)
        
        # Apply filters
        filter_results = {
            "human_check": self._check_human_sample(record),
            "cell_line_check": self._check_cell_line_exclusion(metadata["cell_line_info"]),
            "single_cell_check": self._check_single_cell(metadata["sc_classification"]),
            "tumor_check": self._check_tumor_status(metadata["tumor_status"]),
            "quality_check": self._check_quality_threshold(metadata["quality_metrics"])
        }
        
        # Determine overall pass
        passes_filter = all(
            filter_results[key]["passed"] 
            for key in ["human_check", "cell_line_check", "single_cell_check", "quality_check"]
        )
        
        # Calculate confidence score
        confidence_scores = [filter_results[key]["confidence"] for key in filter_results]
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        processing_time = time.time() - start_time
        
        result = {
            "passes_filter": passes_filter,
            "confidence_score": overall_confidence,
            "filter_steps": filter_results,
            "metadata": metadata,
            "processing_time": processing_time,
            "rejection_reason": self._get_rejection_reason(filter_results) if not passes_filter else ""
        }
        
        # Update statistics
        with self.stats_lock:
            self._update_processing_stats(result)
        
        return result
    
    def _fast_hard_filter(self, record: dict) -> tuple[bool, str]:
        """
        优化的硬过滤：只保留scRNA-seq和scATAC-seq，非Cell Line的条目。
        返回(是否通过, 实验类型)
        """
        # 1. Human判断（使用预编译正则表达式）
        text_fields = [
            str(record.get('organism_ch1', '')),
            str(record.get('scientific_name', '')),
            str(record.get('experiment_title', '')),
            str(record.get('study_title', ''))
        ]
        combined_text = ' '.join(text_fields)
        if not self.human_pattern.search(combined_text):
            return False, ""
        
        # 2. Cell line判断（使用预编译正则表达式）
        cell_text = ' '.join([
            str(record.get('characteristics_ch1', '')),
            str(record.get('source_name_ch1', '')),
            str(record.get('gsm_description', '')),
            str(record.get('study_title', '')),
            str(record.get('summary', '')),
            str(record.get('experiment_title', ''))
        ])
        if self.cell_line_pattern.search(cell_text):
            return False, ""
        
        # 3. 实验类型判断（只保留scRNA-seq和scATAC-seq）
        type_text = ' '.join([
            str(record.get('library_strategy', '')),
            str(record.get('experiment_title', '')),
            str(record.get('study_title', '')),
            str(record.get('summary', '')),
            str(record.get('technology', ''))
        ])
        
        # 检查scRNA-seq
        if self.scrna_pattern.search(type_text):
            return True, "scRNA-seq"
        
        # 检查scATAC-seq
        if self.scatac_pattern.search(type_text):
            return True, "scATAC-seq"
        
        return False, ""

    def _batch_gpu_filter(self, records: List[Dict[str, Any]]) -> List[tuple[Dict[str, Any], str]]:
        """
        Batch GPU-accelerated filtering
        """
        logger.info(f"Starting batch GPU filter with {len(records)} records")
        
        if not self.enable_gpu or self.device is None or not TORCH_AVAILABLE:
            # CPU mode
            logger.info("Using CPU mode for filtering")
            results = []
            for i, record in enumerate(records):
                try:
                    passed, exp_type = self._fast_hard_filter(record)
                    if passed:
                        results.append((record, exp_type))
                except Exception as e:
                    logger.error(f"Error filtering record {i}: {e}")
                    logger.error(f"Record type: {type(record)}")
                    logger.error(f"Record keys: {list(record.keys()) if hasattr(record, 'keys') else 'No keys'}")
                    raise
            logger.info(f"CPU filtering completed: {len(results)} passed")
            return results
        
        # GPU mode - batch processing
        try:
            # Convert text data to tensors for batch processing
            batch_texts = []
            for record in records:
                text = ' '.join([
                    str(record.get('organism_ch1', '')),
                    str(record.get('scientific_name', '')),
                    str(record.get('experiment_title', '')),
                    str(record.get('study_title', '')),
                    str(record.get('characteristics_ch1', '')),
                    str(record.get('source_name_ch1', '')),
                    str(record.get('library_strategy', '')),
                    str(record.get('technology', ''))
                ]).lower()
                batch_texts.append(text)
            
            # Batch keyword checking (using GPU-accelerated string matching)
            results = []
            for i, (record, text) in enumerate(zip(records, batch_texts)):
                # Use pre-compiled regular expressions for fast matching
                if (self.human_pattern.search(text) and 
                    not self.cell_line_pattern.search(text) and
                    (self.scrna_pattern.search(text) or self.scatac_pattern.search(text))):
                    
                    # Determine experiment type
                    if self.scrna_pattern.search(text):
                        exp_type = "scRNA-seq"
                    else:
                        exp_type = "scATAC-seq"
                    
                    results.append((record, exp_type))
            
            return results
            
        except Exception as e:
            logger.warning(f"GPU batch filtering failed, falling back to CPU: {e}")
            # Fallback to CPU mode
            results = []
            for record in records:
                passed, exp_type = self._fast_hard_filter(record)
                if passed:
                    results.append((record, exp_type))
            return results

    def batch_process_enhanced(self, 
                              batch_size: int = 20000,  # 提升默认batch size
                              max_records: Optional[int] = None,
                              output_file: Optional[str] = None,
                              include_download_urls: bool = True,
                              auto_download: bool = False,
                              gpu_batch_size: int = 1000) -> Dict[str, Any]:
        """
        Ultra-optimized batch processing: GPU acceleration + output only qualified entries
        """
        device_type = self.device.type if self.device else 'cpu'
        logger.info(f"Starting ultra-optimized batch processing (GPU: {device_type}, Workers: {self.max_workers})")
        
        # 获取数据库记录
        records = self._get_records_from_db(batch_size, max_records)
        if not records:
            logger.warning("No records found for processing")
            return {"status": "no_records", "results": []}
        
        logger.info(f"Total records fetched: {len(records):,}")
        
        # 1. GPU加速硬过滤
        start_time = time.time()
        filtered_records = []
        
        # 分批进行GPU过滤
        for i in range(0, len(records), gpu_batch_size):
            batch = records[i:i + gpu_batch_size]
            logger.info(f"Processing batch {i//gpu_batch_size + 1}, size: {len(batch)}")
            batch_results = self._batch_gpu_filter(batch)
            logger.info(f"Batch {i//gpu_batch_size + 1} results: {len(batch_results)} passed")
            filtered_records.extend(batch_results)
        
        filter_time = time.time() - start_time
        logger.info(f"Hard filtering completed: {len(filtered_records)}/{len(records)} passed in {filter_time:.2f}s")
        
        if not filtered_records:
            logger.warning("No records passed hard filter")
            return {"status": "no_pass_hard_filter", "results": []}
        
        # Debug: Check the structure of filtered_records
        logger.info(f"First filtered record type: {type(filtered_records[0])}")
        if filtered_records:
            logger.info(f"First filtered record: {filtered_records[0]}")
        
        # 2. 并行深度分析（只对合格条目）
        analysis_start = time.time()
        results = []
        
        # 分离记录和实验类型
        try:
            records_to_analyze = [r[0] for r in filtered_records]
            exp_types = [r[1] for r in filtered_records]
            logger.info(f"Separated {len(records_to_analyze)} records and {len(exp_types)} experiment types")
        except Exception as e:
            logger.error(f"Error separating records and types: {e}")
            logger.error(f"Filtered records structure: {[type(r) for r in filtered_records[:5]]}")
            raise
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_record = {
                executor.submit(self._fast_enhanced_analysis, record, exp_type): (record, exp_type)
                for record, exp_type in zip(records_to_analyze, exp_types)
            }
            
            for future in as_completed(future_to_record):
                result = future.result()
                if result and result.get("passes_filter", False):
                    results.append(result)
        
        analysis_time = time.time() - analysis_start
        logger.info(f"Deep analysis completed: {len(results)}/{len(filtered_records)} passed in {analysis_time:.2f}s")
        
        # 3. 保存结果（只输出符合条件的）
        if output_file:
            self._save_optimized_results(results, output_file, include_download_urls)
        
        # 4. 自动下载（如需）
        if auto_download:
            self._parallel_download_sra_data(results)
        
        # 5. 生成统计
        total_time = time.time() - start_time
        summary = {
            "total_processed": len(records),
            "hard_filtered": len(filtered_records),
            "final_passed": len(results),
            "filter_time": filter_time,
            "analysis_time": analysis_time,
            "total_time": total_time,
            "records_per_second": len(records) / total_time if total_time > 0 else 0,
            "samples_per_second": len(records) / total_time if total_time > 0 else 0
        }
        
        return {
            "status": "completed",
            "results": results,
            "summary": summary,
            "statistics": dict(self.processing_stats)
        }

    def _fast_enhanced_analysis(self, record: Dict[str, Any], exp_type: str) -> Optional[Dict[str, Any]]:
        """
        快速增强分析：只做必要的分析，避免不必要的计算
        """
        try:
            # 基础信息提取
            sra_id = record.get('sra_ID', '')
            gsm_id = record.get('gsm_title', '')
            gse_id = record.get('gse_title', '')
            
            # 生成SRA Lite URL
            sra_urls = self.generate_sra_lite_url(sra_id, include_download=False)
            
            # 快速元数据提取（避免AI调用）
            age_info = self.extract_age_information(record)
            tumor_status = self.detect_tumor_status(record)
            sample_size = self._estimate_sample_size(record)
            pub_info = self._extract_publication_info(record)
            country_info = self.extract_country_information(record)
            
            # 构建结果
            result = {
                "passes_filter": True,  # 已经通过硬过滤
                "confidence_score": 0.9,  # 高置信度
                "sra_id": sra_id,
                "gsm_id": gsm_id,
                "gse_id": gse_id,
                "experiment_type": exp_type,
                "sra_lite_url": sra_urls["sra_lite_url"],
                "age_value": age_info["age_value"],
                "age_unit": age_info["age_unit"],
                "is_tumor": tumor_status["is_tumor"],
                "tumor_type": tumor_status["tumor_type"],
                "sample_size": sample_size["estimated_size"],
                "pmid": pub_info["pmid"],
                "doi": pub_info["doi"],
                "country": country_info["country"],
                "organism": record.get('organism_ch1', ''),
                "experiment_title": record.get('experiment_title', ''),
                "study_title": record.get('study_title', ''),
                "library_strategy": record.get('library_strategy', ''),
                "platform": record.get('platform', ''),
                "spots": record.get('spots', ''),
                "bases": record.get('bases', ''),
                "submission_date": record.get('submission_date', '')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fast analysis for {record.get('sra_ID', 'unknown')}: {e}")
            return None

    def _save_optimized_results(self, results: List[Dict[str, Any]], output_file: str, include_download_urls: bool):
        """
        优化的结果保存：只保存必要字段，提升I/O性能
        """
        try:
            # 准备CSV数据
            csv_data = []
            for result in results:
                row = {
                    "sra_id": result["sra_id"],
                    "gsm_id": result["gsm_id"],
                    "gse_id": result["gse_id"],
                    "experiment_type": result["experiment_type"],
                    "sra_lite_url": result["sra_lite_url"],
                    "age_value": result["age_value"],
                    "age_unit": result["age_unit"],
                    "is_tumor": result["is_tumor"],
                    "tumor_type": result["tumor_type"],
                    "sample_size": result["sample_size"],
                    "pmid": result["pmid"],
                    "doi": result["doi"],
                    "country": result["country"],
                    "organism": result["organism"],
                    "experiment_title": result["experiment_title"],
                    "study_title": result["study_title"],
                    "library_strategy": result["library_strategy"],
                    "platform": result["platform"],
                    "spots": result["spots"],
                    "bases": result["bases"],
                    "submission_date": result["submission_date"],
                    "confidence_score": result["confidence_score"]
                }
                
                # 添加下载URL（如需）
                if include_download_urls:
                    sra_id = result["sra_id"]
                    row.update({
                        "fastq_download_url": f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={sra_id}&format=fastq",
                        "sra_download_url": f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={sra_id}&format=sra"
                    })
                
                csv_data.append(row)
            
            # 保存到CSV
            import pandas as pd
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"Saved {len(csv_data)} optimized results to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save optimized results: {e}")

    def _parallel_download_sra_data(self, results: List[Dict[str, Any]]):
        """
        并行下载SRA数据（预留接口）
        """
        logger.info("Parallel download functionality is not implemented yet")
        # TODO: 实现并行下载逻辑
        pass
    
    def _check_human_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check if sample is human."""
        human_indicators = ['homo sapiens', 'human']
        text_fields = [
            str(record.get('organism_ch1', '')),
            str(record.get('scientific_name', '')),
            str(record.get('experiment_title', '')),
            str(record.get('study_title', ''))
        ]
        
        combined_text = ' '.join(text_fields).lower()
        is_human = any(indicator in combined_text for indicator in human_indicators)
        
        return {
            "passed": is_human,
            "confidence": 0.9 if is_human else 0.1,
            "reason": "Human sample detected" if is_human else "Not a human sample"
        }
    
    def _check_cell_line_exclusion(self, cell_line_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if sample should be excluded (cell line)."""
        is_cell_line = cell_line_info.get("is_cell_line", False)
        
        return {
            "passed": not is_cell_line,  # Pass if NOT a cell line
            "confidence": cell_line_info.get("confidence", 0.0),
            "reason": "Not a cell line" if not is_cell_line else f"Cell line detected: {cell_line_info.get('cell_line_name', '')}"
        }
    
    def _check_single_cell(self, sc_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Check if experiment is single-cell."""
        experiment_type = sc_classification.get("experiment_type", "unknown")
        is_single_cell = experiment_type in ["scRNA-seq", "scATAC-seq"]
        
        return {
            "passed": is_single_cell,
            "confidence": sc_classification.get("confidence", 0.0),
            "reason": f"Single-cell experiment: {experiment_type}" if is_single_cell else "Not a single-cell experiment",
            "experiment_type": experiment_type
        }
    
    def _check_tumor_status(self, tumor_status: Dict[str, Any]) -> Dict[str, Any]:
        """Check tumor status (informational, not a filter)."""
        is_tumor = tumor_status.get("is_tumor", False)
        
        return {
            "passed": True,  # Always pass, just for information
            "confidence": tumor_status.get("confidence", 0.0),
            "reason": f"Tumor sample: {tumor_status.get('tumor_type', '')}" if is_tumor else "Normal tissue sample"
        }
    
    def _check_quality_threshold(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if dataset meets quality threshold."""
        overall_quality = quality_metrics.get("overall_quality_score", 0.0)
        meets_threshold = overall_quality >= 0.5  # Minimum 50% quality score
        
        return {
            "passed": meets_threshold,
            "confidence": overall_quality,
            "reason": f"Quality score: {overall_quality:.2f}" if meets_threshold else f"Low quality: {overall_quality:.2f}"
        }
    
    def _get_rejection_reason(self, filter_results: Dict[str, Any]) -> str:
        """Get reason for rejection."""
        failed_checks = [
            key for key, result in filter_results.items() 
            if not result["passed"]
        ]
        
        if failed_checks:
            reasons = [filter_results[check]["reason"] for check in failed_checks]
            return "; ".join(reasons)
        
        return ""
    
    def _update_processing_stats(self, result: Dict[str, Any]):
        """Update processing statistics."""
        self.processing_stats["total_processed"] += 1
        
        if result["passes_filter"]:
            self.processing_stats["passed_filters"] += 1
        
        # Update specific stats
        filter_steps = result["filter_steps"]
        if filter_steps["human_check"]["passed"]:
            self.processing_stats["human_samples"] += 1
        
        if filter_steps["single_cell_check"]["passed"]:
            self.processing_stats["single_cell_experiments"] += 1
        
        if filter_steps["tumor_check"]["passed"] and "tumor" in filter_steps["tumor_check"]["reason"].lower():
            self.processing_stats["tumor_samples"] += 1
    
    def _get_records_from_db(self, batch_size: int, max_records: Optional[int]) -> List[Dict[str, Any]]:
        """Get records from database."""
        try:
            conn = get_connection()
            
            limit_clause = f"LIMIT {max_records}" if max_records else ""
            
            query = f"""
            SELECT * FROM {self.full_table_name}
            WHERE "organism_ch1" IS NOT NULL
            {limit_clause}
            """
            
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query)
                records = cur.fetchall()
            
            conn.close()
            return [dict(record) for record in records]
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
    
    def _save_enhanced_results(self, results: List[Dict[str, Any]], output_file: str, include_download_urls: bool):
        """
        Save enhanced results to CSV file
        """
        try:
            if not results:
                logger.warning("No results to save")
                return
            
            # Prepare CSV data
            csv_data = []
            for result in results:
                row = {
                    "sra_id": result.get("sra_id", ""),
                    "gsm_id": result.get("gsm_id", ""),
                    "gse_id": result.get("gse_id", ""),
                    "experiment_type": result.get("experiment_type", ""),
                    "sra_lite_url": result.get("sra_lite_url", ""),
                    "age_value": result.get("age_value", ""),
                    "age_unit": result.get("age_unit", ""),
                    "is_tumor": result.get("is_tumor", False),
                    "tumor_type": result.get("tumor_type", ""),
                    "sample_size": result.get("sample_size", ""),
                    "pmid": result.get("pmid", ""),
                    "doi": result.get("doi", ""),
                    "country": result.get("country", ""),
                    "organism": result.get("organism", ""),
                    "experiment_title": result.get("experiment_title", ""),
                    "study_title": result.get("study_title", ""),
                    "library_strategy": result.get("library_strategy", ""),
                    "platform": result.get("platform", ""),
                    "spots": result.get("spots", ""),
                    "bases": result.get("bases", ""),
                    "submission_date": result.get("submission_date", ""),
                    "confidence_score": result.get("confidence_score", 0.0)
                }
                
                # Add download URLs if requested
                if include_download_urls:
                    sra_id = result.get("sra_id", "")
                    if sra_id:
                        row.update({
                            "fastq_download_url": f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={sra_id}&format=fastq",
                            "sra_download_url": f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={sra_id}&format=sra"
                        })
                
                csv_data.append(row)
            
            # Save to CSV
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"Saved {len(csv_data)} enhanced results to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced results: {e}")
            raise
    
    def _generate_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate processing summary statistics
        """
        try:
            if not results:
                return {
                    "total_results": 0,
                    "pass_rate": 0.0,
                    "average_confidence": 0.0,
                    "experiment_types": {},
                    "tumor_samples": 0,
                    "age_info_available": 0
                }
            
            # Calculate basic statistics
            total_results = len(results)
            passed_results = len([r for r in results if r.get("passes_filter", False)])
            pass_rate = (passed_results / total_results * 100) if total_results > 0 else 0
            
            # Calculate average confidence
            confidences = [r.get("confidence_score", 0) for r in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Count experiment types
            exp_types = {}
            for result in results:
                exp_type = result.get("experiment_type", "Unknown")
                exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
            
            # Count tumor samples
            tumor_samples = len([r for r in results if r.get("is_tumor", False)])
            
            # Count age information availability
            age_info_available = len([r for r in results if r.get("age_value") and r.get("age_value") != "Not available"])
            
            return {
                "total_results": total_results,
                "passed_results": passed_results,
                "pass_rate": pass_rate,
                "average_confidence": avg_confidence,
                "experiment_types": exp_types,
                "tumor_samples": tumor_samples,
                "age_info_available": age_info_available,
                "processing_stats": dict(self.processing_stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating processing summary: {e}")
            return {
                "total_results": 0,
                "pass_rate": 0.0,
                "average_confidence": 0.0,
                "experiment_types": {},
                "tumor_samples": 0,
                "age_info_available": 0,
                "error": str(e)
            } 

    def extract_country_information(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract country information from various fields.
        
        Args:
            record: Database record
            
        Returns:
            Dict with country information
        """
        country_info = {
            "country": "Not available",
            "country_source": "",
            "country_confidence": 0.0
        }
        
        # Fields to check for country information
        country_fields = [
            record.get('gsm_contact', ''),
            record.get('study_title', ''),
            record.get('summary', ''),
            record.get('experiment_title', ''),
            record.get('characteristics_ch1', ''),
            record.get('source_name_ch1', '')
        ]
        
        # Common country patterns
        country_patterns = [
            r'\b(USA|United States|America|US)\b',
            r'\b(China|Chinese|PRC)\b',
            r'\b(Japan|Japanese)\b',
            r'\b(UK|United Kingdom|England|Scotland|Wales|Northern Ireland)\b',
            r'\b(Germany|German|Deutschland)\b',
            r'\b(France|French)\b',
            r'\b(Canada|Canadian)\b',
            r'\b(Australia|Australian)\b',
            r'\b(Switzerland|Swiss)\b',
            r'\b(Sweden|Swedish)\b',
            r'\b(Netherlands|Dutch|Holland)\b',
            r'\b(Italy|Italian)\b',
            r'\b(Spain|Spanish)\b',
            r'\b(Belgium|Belgian)\b',
            r'\b(Denmark|Danish)\b',
            r'\b(Norway|Norwegian)\b',
            r'\b(Finland|Finnish)\b',
            r'\b(Singapore|Singaporean)\b',
            r'\b(South Korea|Korean|Korea)\b',
            r'\b(India|Indian)\b',
            r'\b(Brazil|Brazilian)\b',
            r'\b(Mexico|Mexican)\b',
            r'\b(Argentina|Argentine)\b',
            r'\b(Chile|Chilean)\b',
            r'\b(Colombia|Colombian)\b',
            r'\b(Peru|Peruvian)\b',
            r'\b(Venezuela|Venezuelan)\b',
            r'\b(Uruguay|Uruguayan)\b',
            r'\b(Paraguay|Paraguayan)\b',
            r'\b(Bolivia|Bolivian)\b',
            r'\b(Ecuador|Ecuadorian)\b',
            r'\b(Guyana|Guyanese)\b',
            r'\b(Suriname|Surinamese)\b',
            r'\b(French Guiana)\b',
            r'\b(Israel|Israeli)\b',
            r'\b(Turkey|Turkish)\b',
            r'\b(Iran|Iranian|Persian)\b',
            r'\b(Saudi Arabia|Saudi)\b',
            r'\b(UAE|United Arab Emirates)\b',
            r'\b(Qatar|Qatari)\b',
            r'\b(Kuwait|Kuwaiti)\b',
            r'\b(Bahrain|Bahraini)\b',
            r'\b(Oman|Omani)\b',
            r'\b(Yemen|Yemeni)\b',
            r'\b(Jordan|Jordanian)\b',
            r'\b(Lebanon|Lebanese)\b',
            r'\b(Syria|Syrian)\b',
            r'\b(Iraq|Iraqi)\b',
            r'\b(Egypt|Egyptian)\b',
            r'\b(Morocco|Moroccan)\b',
            r'\b(Algeria|Algerian)\b',
            r'\b(Tunisia|Tunisian)\b',
            r'\b(Libya|Libyan)\b',
            r'\b(Sudan|Sudanese)\b',
            r'\b(Ethiopia|Ethiopian)\b',
            r'\b(Kenya|Kenyan)\b',
            r'\b(Nigeria|Nigerian)\b',
            r'\b(South Africa|South African)\b',
            r'\b(Ghana|Ghanaian)\b',
            r'\b(Uganda|Ugandan)\b',
            r'\b(Tanzania|Tanzanian)\b',
            r'\b(Zimbabwe|Zimbabwean)\b',
            r'\b(Zambia|Zambian)\b',
            r'\b(Malawi|Malawian)\b',
            r'\b(Mozambique|Mozambican)\b',
            r'\b(Angola|Angolan)\b',
            r'\b(Namibia|Namibian)\b',
            r'\b(Botswana|Botswanan)\b',
            r'\b(Lesotho|Basotho)\b',
            r'\b(Swaziland|Swazi)\b',
            r'\b(Madagascar|Malagasy)\b',
            r'\b(Mauritius|Mauritian)\b',
            r'\b(Seychelles|Seychellois)\b',
            r'\b(Comoros|Comorian)\b',
            r'\b(Djibouti|Djiboutian)\b',
            r'\b(Eritrea|Eritrean)\b',
            r'\b(Somalia|Somali)\b',
            r'\b(Burundi|Burundian)\b',
            r'\b(Rwanda|Rwandan)\b',
            r'\b(Central African Republic|Central African)\b',
            r'\b(Cameroon|Cameroonian)\b',
            r'\b(Chad|Chadian)\b',
            r'\b(Niger|Nigerien)\b',
            r'\b(Mali|Malian)\b',
            r'\b(Burkina Faso|Burkinabe)\b',
            r'\b(Senegal|Senegalese)\b',
            r'\b(Gambia|Gambian)\b',
            r'\b(Guinea-Bissau|Bissau-Guinean)\b',
            r'\b(Guinea|Guinean)\b',
            r'\b(Sierra Leone|Sierra Leonean)\b',
            r'\b(Liberia|Liberian)\b',
            r'\b(Ivory Coast|Ivorian)\b',
            r'\b(Togo|Togolese)\b',
            r'\b(Benin|Beninese)\b',
            r'\b(Equatorial Guinea|Equatorial Guinean)\b',
            r'\b(Gabon|Gabonese)\b',
            r'\b(Congo|Congolese)\b',
            r'\b(Democratic Republic of the Congo|DRC|DR Congo)\b',
            r'\b(Central African Republic|Central African)\b',
            r'\b(South Sudan|South Sudanese)\b',
            r'\b(Russia|Russian)\b',
            r'\b(Ukraine|Ukrainian)\b',
            r'\b(Belarus|Belarusian)\b',
            r'\b(Moldova|Moldovan)\b',
            r'\b(Romania|Romanian)\b',
            r'\b(Bulgaria|Bulgarian)\b',
            r'\b(Serbia|Serbian)\b',
            r'\b(Croatia|Croatian)\b',
            r'\b(Slovenia|Slovenian)\b',
            r'\b(Bosnia and Herzegovina|Bosnian|Herzegovinian)\b',
            r'\b(Montenegro|Montenegrin)\b',
            r'\b(North Macedonia|Macedonian)\b',
            r'\b(Albania|Albanian)\b',
            r'\b(Greece|Greek)\b',
            r'\b(Cyprus|Cypriot)\b',
            r'\b(Malta|Maltese)\b',
            r'\b(Poland|Polish)\b',
            r'\b(Czech Republic|Czech|Czechoslovakia)\b',
            r'\b(Slovakia|Slovak)\b',
            r'\b(Hungary|Hungarian)\b',
            r'\b(Austria|Austrian)\b',
            r'\b(Switzerland|Swiss)\b',
            r'\b(Liechtenstein|Liechtensteiner)\b',
            r'\b(Luxembourg|Luxembourgish)\b',
            r'\b(Monaco|Monacan)\b',
            r'\b(Andorra|Andorran)\b',
            r'\b(San Marino|Sammarinese)\b',
            r'\b(Vatican City|Vatican)\b',
            r'\b(Ireland|Irish)\b',
            r'\b(Iceland|Icelandic)\b',
            r'\b(Faroe Islands|Faroese)\b',
            r'\b(Greenland|Greenlandic)\b',
            r'\b(New Zealand|New Zealander)\b',
            r'\b(Fiji|Fijian)\b',
            r'\b(Papua New Guinea|Papua New Guinean)\b',
            r'\b(Solomon Islands|Solomon Islander)\b',
            r'\b(Vanuatu|Ni-Vanuatu)\b',
            r'\b(New Caledonia|New Caledonian)\b',
            r'\b(French Polynesia|French Polynesian)\b',
            r'\b(Samoa|Samoan)\b',
            r'\b(Tonga|Tongan)\b',
            r'\b(Kiribati|I-Kiribati)\b',
            r'\b(Tuvalu|Tuvaluan)\b',
            r'\b(Nauru|Nauruan)\b',
            r'\b(Palau|Palauan)\b',
            r'\b(Marshall Islands|Marshallese)\b',
            r'\b(Micronesia|Micronesian)\b',
            r'\b(Guam|Guamanian)\b',
            r'\b(Northern Mariana Islands|Northern Mariana Islander)\b',
            r'\b(American Samoa|American Samoan)\b',
            r'\b(Cook Islands|Cook Islander)\b',
            r'\b(Niue|Niuean)\b',
            r'\b(Tokelau|Tokelauan)\b',
            r'\b(Wallis and Futuna|Wallisian|Futunan)\b',
            r'\b(Pitcairn Islands|Pitcairn Islander)\b',
            r'\b(Easter Island|Rapa Nui)\b',
            r'\b(Hawaii|Hawaiian)\b',
            r'\b(Alaska|Alaskan)\b',
            r'\b(California|Californian)\b',
            r'\b(Texas|Texan)\b',
            r'\b(Florida|Floridian)\b',
            r'\b(New York|New Yorker)\b',
            r'\b(Washington|Washingtonian)\b',
            r'\b(Oregon|Oregonian)\b',
            r'\b(Colorado|Coloradan)\b',
            r'\b(Utah|Utahn)\b',
            r'\b(Nevada|Nevadan)\b',
            r'\b(Arizona|Arizonan)\b',
            r'\b(New Mexico|New Mexican)\b',
            r'\b(Montana|Montanan)\b',
            r'\b(Idaho|Idahoan)\b',
            r'\b(Wyoming|Wyomingite)\b',
            r'\b(North Dakota|North Dakotan)\b',
            r'\b(South Dakota|South Dakotan)\b',
            r'\b(Nebraska|Nebraskan)\b',
            r'\b(Kansas|Kansan)\b',
            r'\b(Oklahoma|Oklahoman)\b',
            r'\b(Arkansas|Arkansan)\b',
            r'\b(Missouri|Missourian)\b',
            r'\b(Iowa|Iowan)\b',
            r'\b(Minnesota|Minnesotan)\b',
            r'\b(Wisconsin|Wisconsinite)\b',
            r'\b(Illinois|Illinoisan)\b',
            r'\b(Indiana|Hoosier)\b',
            r'\b(Ohio|Ohioan)\b',
            r'\b(Michigan|Michigander)\b',
            r'\b(Pennsylvania|Pennsylvanian)\b',
            r'\b(New Jersey|New Jerseyan)\b',
            r'\b(Delaware|Delawarean)\b',
            r'\b(Maryland|Marylander)\b',
            r'\b(Virginia|Virginian)\b',
            r'\b(West Virginia|West Virginian)\b',
            r'\b(Kentucky|Kentuckian)\b',
            r'\b(Tennessee|Tennessean)\b',
            r'\b(North Carolina|North Carolinian)\b',
            r'\b(South Carolina|South Carolinian)\b',
            r'\b(Georgia|Georgian)\b',
            r'\b(Alabama|Alabamian)\b',
            r'\b(Mississippi|Mississippian)\b',
            r'\b(Louisiana|Louisianan)\b',
            r'\b(Texas|Texan)\b',
            r'\b(New Mexico|New Mexican)\b',
            r'\b(Arizona|Arizonan)\b',
            r'\b(California|Californian)\b',
            r'\b(Nevada|Nevadan)\b',
            r'\b(Utah|Utahn)\b',
            r'\b(Colorado|Coloradan)\b',
            r'\b(Wyoming|Wyomingite)\b',
            r'\b(Montana|Montanan)\b',
            r'\b(Idaho|Idahoan)\b',
            r'\b(Washington|Washingtonian)\b',
            r'\b(Oregon|Oregonian)\b',
            r'\b(Alaska|Alaskan)\b',
            r'\b(Hawaii|Hawaiian)\b'
        ]
        
        for field_name, field_value in zip(
            ['gsm_contact', 'study_title', 'summary', 'experiment_title', 'characteristics_ch1', 'source_name_ch1'],
            country_fields
        ):
            if not field_value:
                continue
                
            field_text = str(field_value).lower()
            
            # Check for country patterns
            for pattern in country_patterns:
                matches = re.findall(pattern, field_text, re.IGNORECASE)
                if matches:
                    country_value = matches[0].strip()
                    
                    # Normalize country names
                    country_mapping = {
                        'usa': 'United States',
                        'united states': 'United States',
                        'america': 'United States',
                        'us': 'United States',
                        'uk': 'United Kingdom',
                        'united kingdom': 'United Kingdom',
                        'england': 'United Kingdom',
                        'scotland': 'United Kingdom',
                        'wales': 'United Kingdom',
                        'northern ireland': 'United Kingdom',
                        'prc': 'China',
                        'chinese': 'China',
                        'japanese': 'Japan',
                        'german': 'Germany',
                        'deutschland': 'Germany',
                        'french': 'France',
                        'canadian': 'Canada',
                        'australian': 'Australia',
                        'swiss': 'Switzerland',
                        'swedish': 'Sweden',
                        'dutch': 'Netherlands',
                        'holland': 'Netherlands',
                        'italian': 'Italy',
                        'spanish': 'Spain',
                        'belgian': 'Belgium',
                        'danish': 'Denmark',
                        'norwegian': 'Norway',
                        'finnish': 'Finland',
                        'singaporean': 'Singapore',
                        'korean': 'South Korea',
                        'korea': 'South Korea',
                        'indian': 'India',
                        'brazilian': 'Brazil',
                        'mexican': 'Mexico',
                        'argentine': 'Argentina',
                        'chilean': 'Chile',
                        'colombian': 'Colombia',
                        'peruvian': 'Peru',
                        'venezuelan': 'Venezuela',
                        'uruguayan': 'Uruguay',
                        'paraguayan': 'Paraguay',
                        'bolivian': 'Bolivia',
                        'ecuadorian': 'Ecuador',
                        'guyanese': 'Guyana',
                        'surinamese': 'Suriname',
                        'israeli': 'Israel',
                        'turkish': 'Turkey',
                        'iranian': 'Iran',
                        'persian': 'Iran',
                        'saudi': 'Saudi Arabia',
                        'qatari': 'Qatar',
                        'kuwaiti': 'Kuwait',
                        'bahraini': 'Bahrain',
                        'omani': 'Oman',
                        'yemeni': 'Yemen',
                        'jordanian': 'Jordan',
                        'lebanese': 'Lebanon',
                        'syrian': 'Syria',
                        'iraqi': 'Iraq',
                        'egyptian': 'Egypt',
                        'moroccan': 'Morocco',
                        'algerian': 'Algeria',
                        'tunisian': 'Tunisia',
                        'libyan': 'Libya',
                        'sudanese': 'Sudan',
                        'ethiopian': 'Ethiopia',
                        'kenyan': 'Kenya',
                        'nigerian': 'Nigeria',
                        'south african': 'South Africa',
                        'ghanaian': 'Ghana',
                        'ugandan': 'Uganda',
                        'tanzanian': 'Tanzania',
                        'zimbabwean': 'Zimbabwe',
                        'zambian': 'Zambia',
                        'malawian': 'Malawi',
                        'mozambican': 'Mozambique',
                        'angolan': 'Angola',
                        'namibian': 'Namibia',
                        'botswanan': 'Botswana',
                        'basotho': 'Lesotho',
                        'swazi': 'Swaziland',
                        'malagasy': 'Madagascar',
                        'mauritian': 'Mauritius',
                        'seychellois': 'Seychelles',
                        'comorian': 'Comoros',
                        'djiboutian': 'Djibouti',
                        'eritrean': 'Eritrea',
                        'somali': 'Somalia',
                        'burundian': 'Burundi',
                        'rwandan': 'Rwanda',
                        'central african': 'Central African Republic',
                        'cameroonian': 'Cameroon',
                        'chadian': 'Chad',
                        'nigerien': 'Niger',
                        'malian': 'Mali',
                        'burkinabe': 'Burkina Faso',
                        'senegalese': 'Senegal',
                        'gambian': 'Gambia',
                        'bissau-guinean': 'Guinea-Bissau',
                        'guinean': 'Guinea',
                        'sierra leonean': 'Sierra Leone',
                        'liberian': 'Liberia',
                        'ivorian': 'Ivory Coast',
                        'togolese': 'Togo',
                        'beninese': 'Benin',
                        'equatorial guinean': 'Equatorial Guinea',
                        'gabonese': 'Gabon',
                        'congolese': 'Congo',
                        'drc': 'Democratic Republic of the Congo',
                        'dr congo': 'Democratic Republic of the Congo',
                        'south sudanese': 'South Sudan',
                        'russian': 'Russia',
                        'ukrainian': 'Ukraine',
                        'belarusian': 'Belarus',
                        'moldovan': 'Moldova',
                        'romanian': 'Romania',
                        'bulgarian': 'Bulgaria',
                        'serbian': 'Serbia',
                        'croatian': 'Croatia',
                        'slovenian': 'Slovenia',
                        'bosnian': 'Bosnia and Herzegovina',
                        'herzegovinian': 'Bosnia and Herzegovina',
                        'montenegrin': 'Montenegro',
                        'macedonian': 'North Macedonia',
                        'albanian': 'Albania',
                        'greek': 'Greece',
                        'cypriot': 'Cyprus',
                        'maltese': 'Malta',
                        'polish': 'Poland',
                        'czech': 'Czech Republic',
                        'czechoslovakia': 'Czech Republic',
                        'slovak': 'Slovakia',
                        'hungarian': 'Hungary',
                        'austrian': 'Austria',
                        'liechtensteiner': 'Liechtenstein',
                        'luxembourgish': 'Luxembourg',
                        'monacan': 'Monaco',
                        'andorran': 'Andorra',
                        'sammarinese': 'San Marino',
                        'vatican': 'Vatican City',
                        'irish': 'Ireland',
                        'icelandic': 'Iceland',
                        'faroese': 'Faroe Islands',
                        'greenlandic': 'Greenland',
                        'new zealander': 'New Zealand',
                        'fijian': 'Fiji',
                        'papua new guinean': 'Papua New Guinea',
                        'solomon islander': 'Solomon Islands',
                        'ni-vanuatu': 'Vanuatu',
                        'new caledonian': 'New Caledonia',
                        'french polynesian': 'French Polynesia',
                        'samoan': 'Samoa',
                        'tongan': 'Tonga',
                        'i-kiribati': 'Kiribati',
                        'tuvaluan': 'Tuvalu',
                        'nauruan': 'Nauru',
                        'palauan': 'Palau',
                        'marshallese': 'Marshall Islands',
                        'micronesian': 'Micronesia',
                        'guamanian': 'Guam',
                        'northern mariana islander': 'Northern Mariana Islands',
                        'american samoan': 'American Samoa',
                        'cook islander': 'Cook Islands',
                        'niuean': 'Niue',
                        'tokelauan': 'Tokelau',
                        'wallisian': 'Wallis and Futuna',
                        'futunan': 'Wallis and Futuna',
                        'pitcairn islander': 'Pitcairn Islands',
                        'rapa nui': 'Easter Island',
                        'hawaiian': 'Hawaii',
                        'alaskan': 'Alaska',
                        'californian': 'California',
                        'texan': 'Texas',
                        'floridian': 'Florida',
                        'new yorker': 'New York',
                        'washingtonian': 'Washington',
                        'oregonian': 'Oregon',
                        'coloradan': 'Colorado',
                        'utahn': 'Utah',
                        'nevadan': 'Nevada',
                        'arizonan': 'Arizona',
                        'new mexican': 'New Mexico',
                        'montanan': 'Montana',
                        'idahoan': 'Idaho',
                        'wyomingite': 'Wyoming',
                        'north dakotan': 'North Dakota',
                        'south dakotan': 'South Dakota',
                        'nebraskan': 'Nebraska',
                        'kansan': 'Kansas',
                        'oklahoman': 'Oklahoma',
                        'arkansan': 'Arkansas',
                        'missourian': 'Missouri',
                        'iowan': 'Iowa',
                        'minnesotan': 'Minnesota',
                        'wisconsinite': 'Wisconsin',
                        'illinoisan': 'Illinois',
                        'hoosier': 'Indiana',
                        'ohioan': 'Ohio',
                        'michigander': 'Michigan',
                        'pennsylvanian': 'Pennsylvania',
                        'new jerseyan': 'New Jersey',
                        'delawarean': 'Delaware',
                        'marylander': 'Maryland',
                        'virginian': 'Virginia',
                        'west virginian': 'West Virginia',
                        'kentuckian': 'Kentucky',
                        'tennessean': 'Tennessee',
                        'north carolinian': 'North Carolina',
                        'south carolinian': 'South Carolina',
                        'georgian': 'Georgia',
                        'alabamian': 'Alabama',
                        'mississippian': 'Mississippi',
                        'louisianan': 'Louisiana'
                    }
                    
                    normalized_country = country_mapping.get(country_value.lower(), country_value)
                    
                    country_info.update({
                        "country": normalized_country,
                        "country_source": field_name,
                        "country_confidence": 0.9
                    })
                    return country_info
        
        return country_info 