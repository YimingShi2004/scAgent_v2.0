"""
Enhanced filtering system specifically optimized for sra_geo_ft2 table.
Handles 82-column dataset with intelligent human sample identification,
cell line exclusion, and single-cell experiment recognition.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Generator, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .db.connect import get_connection
from .db.merged_table_handler import MergedTableHandler
from .models.client import get_model_client

logger = logging.getLogger(__name__)

class SraGeoFt2Optimizer:
    """
    Specialized optimizer for sra_geo_ft2 table with 82 columns.
    Focuses on human sc-eQTL datasets with intelligent filtering.
    """
    
    def __init__(self, 
                 schema: str = "merged", 
                 table: str = "sra_geo_ft2",
                 ai_client: Optional[Any] = None):
        self.schema = schema
        self.table = table
        self.full_table_name = f'"{schema}"."{table}"'
        self.ai_client = ai_client or get_model_client()
        self.processing_stats = defaultdict(int)
        
        # Column name mapping for sra_geo_ft2 table (基于实际82列结构)
        self.column_mapping = {
            # 核心标识符字段 (基于实际列名)
            "sra_id": "sra_ID",
            "gsm_title": "gsm_title", 
            "gse_title": "gse_title",
            "experiment_title": "experiment_title",
            "run_alias": "run_alias",
            "experiment_alias": "experiment_alias",
            "sample_alias": "sample_alias",
            "study_alias": "study_alias",
            
            # 物种信息字段
            "organism_ch1": "organism_ch1", 
            "organism": "organism",
            "scientific_name": "scientific_name",
            "common_name": "common_name",
            
            # 样本和组织信息
            "characteristics_ch1": "characteristics_ch1",
            "source_name_ch1": "source_name_ch1",
            "sample_attribute": "sample_attribute",
            "description": "description",
            "gsm_description": "gsm_description",
            
            # 研究信息
            "study_title": "study_title",
            "study_abstract": "study_abstract", 
            "gse_title": "gse_title",
            "summary": "summary",
            "overall_design": "overall_design",
            
            # 测序技术信息
            "library_strategy": "library_strategy",
            "library_source": "library_source",
            "library_selection": "library_selection",
            "library_layout": "library_layout",
            "platform": "platform",
            "instrument_model": "instrument_model",
            "technology": "technology",
            
            # 数据量信息
            "spots": "spots",
            "bases": "bases",
            "spot_length": "spot_length",
            
            # 日期信息
            "submission_date": "submission_date",
            "gsm_submission_date": "gsm_submission_date",
            "gse_submission_date": "gse_submission_date",
            "run_date": "run_date",
            
            # 其他重要字段
            "pubmed_id": "pubmed_id",
            "center_project_name": "center_project_name",
            "lab_name": "lab_name"
        }
    
    def is_human_sample_optimized(self, record: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Optimized human sample identification specifically for sra_geo_ft2.
        Adapted for actual table structure with proper NULL handling.
        
        Args:
            record: Database record
            
        Returns:
            Tuple of (is_human, reason, confidence_score)
        """
        human_keywords = ['homo sapiens', 'human', 'h. sapiens', 'hsapiens', 'h.sapiens']
        
        # 1. Priority check: organism_ch1 (most reliable field in this table)
        organism_ch1 = str(record.get('organism_ch1', '') or '').lower().strip()
        if organism_ch1:
            if any(keyword in organism_ch1 for keyword in human_keywords):
                # Handle mixed species: "Homo sapiens; Mus musculus" -> extract human part
                if 'homo sapiens' in organism_ch1:
                    return True, f"Human confirmed in organism_ch1: {organism_ch1}", 0.95
                else:
                    return True, f"Human synonym in organism_ch1: {organism_ch1}", 0.85
            else:
                # Non-human species found
                return False, f"Non-human species in organism_ch1: {organism_ch1}", 0.95
        
        # 2. Check scientific_name field
        scientific_name = str(record.get('scientific_name', '') or '').lower().strip()
        if scientific_name:
            if any(keyword in scientific_name for keyword in human_keywords):
                return True, f"Human found in scientific_name: {scientific_name}", 0.90
            else:
                return False, f"Non-human species in scientific_name: {scientific_name}", 0.90
        
        # 3. Check organism field (often NULL, but check if available)
        organism = str(record.get('organism', '') or '').lower().strip()
        if organism and organism != 'null':
            if any(keyword in organism for keyword in human_keywords):
                return True, f"Human found in organism field: {organism}", 0.75
            else:
                return False, f"Non-human species in organism: {organism}", 0.75
        
        # 4. Check experiment_title if available (often NULL in this table)
        experiment_title = str(record.get('experiment_title', '') or '').lower().strip()
        if experiment_title and experiment_title != 'null':
            # Parse format: "GSM5163384: FSE_RAS_day4_2; Homo sapiens; RNA-Seq"
            title_parts = [part.strip() for part in experiment_title.split(';')]
            for part in title_parts:
                if any(keyword in part for keyword in human_keywords):
                    return True, f"Human found in experiment_title: {part}", 0.80
        
        # 5. Text-based search in study information (fallback)
        study_fields = [
            ('study_title', record.get('study_title', '')),
            ('study_abstract', record.get('study_abstract', '')),
            ('summary', record.get('summary', '')),
            ('gsm_description', record.get('gsm_description', '')),
            ('overall_design', record.get('overall_design', ''))
        ]
        
        for field_name, field_value in study_fields:
            if field_value and str(field_value).strip() and str(field_value).lower() != 'null':
                field_text = str(field_value).lower()
                if any(keyword in field_text for keyword in human_keywords):
                    return True, f"Human indicator in {field_name}: {field_text[:50]}...", 0.60
        
        return False, "No human indicators found", 0.85
    
    def is_cell_line_sample(self, record: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Enhanced cell line detection for sra_geo_ft2.
        Based on actual table structure with proper NULL handling.
        
        Args:
            record: Database record
            
        Returns:
            Tuple of (is_cell_line, reason, confidence_score)
        """
        # Primary check: characteristics_ch1 field (key field in this table)
        characteristics = str(record.get('characteristics_ch1', '') or '').lower().strip()
        if characteristics and characteristics != 'null':
            # Direct cell line indicators
            cell_line_patterns = [
                r'cell\s*line\s*:',
                r'cell\s*line\s*name\s*:',
                r'cell\s*type\s*:\s*cell\s*line',
                r'immortalized\s*cell',
                r'cultured\s*cell\s*line',
                r'transformed\s*cell',
                r'cancer\s*cell\s*line'
            ]
            
            for pattern in cell_line_patterns:
                if re.search(pattern, characteristics):
                    return True, f"Cell line detected in characteristics_ch1: {characteristics[:100]}", 0.95
            
            # Check for common cell line names in characteristics
            cell_line_names = [
                'hek293', '293t', 'hela', 'k562', 'jurkat', 'u87', 'mcf7',
                'a549', 'hct116', 'sw480', 'caco-2', 'h1299', 'pc3', 'cos7',
                'nih3t3', 'cho', 'vero', 'mdck'
            ]
            
            for cell_line in cell_line_names:
                if cell_line in characteristics:
                    return True, f"Known cell line {cell_line} in characteristics_ch1", 0.90
        
        # Secondary check: source_name_ch1 field
        source_name = str(record.get('source_name_ch1', '') or '').lower().strip()
        if source_name and source_name != 'null':
            # Check for cell line indicators in source name
            if 'cell line' in source_name or 'cell_line' in source_name:
                return True, f"Cell line indicator in source_name_ch1: {source_name}", 0.85
            
            # Check for known cell line names
            cell_line_names = [
                'hek293', '293t', 'hela', 'k562', 'jurkat', 'u87', 'mcf7',
                'a549', 'hct116', 'sw480', 'caco-2', 'h1299', 'pc3'
            ]
            
            for cell_line in cell_line_names:
                if cell_line in source_name:
                    return True, f"Known cell line {cell_line} in source_name_ch1", 0.85
        
        # Tertiary check: other text fields
        other_fields = [
            ('gsm_description', record.get('gsm_description', '')),
            ('description', record.get('description', '')),
            ('study_title', record.get('study_title', '')),
            ('experiment_title', record.get('experiment_title', ''))
        ]
        
        for field_name, field_value in other_fields:
            if field_value and str(field_value).strip() and str(field_value).lower() != 'null':
                field_text = str(field_value).lower()
                
                # Check for cell line keywords
                if re.search(r'cell\s*line', field_text):
                    return True, f"Cell line indicator in {field_name}", 0.75
                
                # Check for known cell line names
                cell_line_names = [
                    'hek293', '293t', 'hela', 'k562', 'jurkat', 'u87', 'mcf7',
                    'a549', 'hct116', 'sw480', 'caco-2', 'h1299', 'pc3'
                ]
                
                for cell_line in cell_line_names:
                    if cell_line in field_text:
                        return True, f"Known cell line {cell_line} in {field_name}", 0.75
        
        return False, "No cell line indicators found", 0.90
    
    def is_single_cell_experiment_optimized(self, record: Dict[str, Any]) -> Tuple[bool, str, float, str]:
        """
        Optimized single-cell experiment detection for sra_geo_ft2.
        Adapted for actual table structure where many sequencing fields are NULL.
        
        Args:
            record: Database record
            
        Returns:
            Tuple of (is_single_cell, reason, confidence_score, experiment_type)
        """
        # Single-cell technology keywords (prioritized and expanded)
        sc_technologies = {
            'scRNA-seq': [
                'scrna-seq', 'scrna seq', 'sc-rna-seq', 'sc rna seq', 'sc-rna',
                'single-cell rna', 'single cell rna', 'single-cell rna-seq',
                'single cell rna-seq', 'single-cell transcriptom', 'single cell transcriptom',
                '10x genomics', '10x chromium', 'chromium single cell', '10x',
                'drop-seq', 'dropseq', 'smart-seq', 'smartseq', 'smart seq',
                'cel-seq', 'celseq', 'mars-seq', 'marsseq', 'plate-seq', 'plateseq'
            ],
            'scATAC-seq': [
                'scatac-seq', 'scatac seq', 'sc-atac-seq', 'sc atac seq', 'sc-atac',
                'single-cell atac', 'single cell atac', 'single-cell atac-seq',
                'single cell chromatin', 'single-cell chromatin'
            ],
            'general_sc': [
                'single-cell', 'single cell', 'sc seq', 'sc-seq',
                'in-drop', 'indrop', 'single-cell sequencing', 'single cell sequencing',
                'single-cell analysis', 'single cell analysis'
            ]
        }
        
        # Fields to check (prioritized by data availability in sra_geo_ft2)
        # Based on actual table structure, focus on text fields that have data
        check_fields = [
            ('study_title', record.get('study_title', '')),           # High priority, usually has data
            ('study_abstract', record.get('study_abstract', '')),     # High priority, rich content
            ('summary', record.get('summary', '')),                  # High priority, GSE summary
            ('overall_design', record.get('overall_design', '')),    # High priority, experiment design
            ('gsm_description', record.get('gsm_description', '')),  # Medium priority
            ('experiment_title', record.get('experiment_title', '')), # Often NULL
            ('description', record.get('description', '')),          # Often NULL
            ('library_strategy', record.get('library_strategy', '')), # Often NULL
            ('platform', record.get('platform', '')),               # Often NULL
            ('instrument_model', record.get('instrument_model', '')), # Often NULL
            ('technology', record.get('technology', ''))             # Technology field
        ]
        
        for field_name, field_value in check_fields:
            if not field_value or str(field_value).strip() == '' or str(field_value).lower() == 'null':
                continue
                
            field_text = str(field_value).lower().strip()
            
            # Check each technology category
            for tech_type, keywords in sc_technologies.items():
                for keyword in keywords:
                    if keyword in field_text:
                        # Assign confidence based on field reliability and match quality
                        if field_name in ['study_title', 'study_abstract', 'summary']:
                            confidence = 0.95
                        elif field_name in ['overall_design', 'gsm_description']:
                            confidence = 0.90
                        elif field_name in ['library_strategy', 'technology']:
                            confidence = 0.85
                        else:
                            confidence = 0.80
                            
                        return True, f"Single-cell technology '{keyword}' found in {field_name}", confidence, tech_type
        
        # Special handling for RNA-seq experiments that might be single-cell
        # This is crucial for cases like "GSM4640426: devTOXB01_319C_BPS_400; Homo sapiens; RNA-Seq"
        text_fields = [
            record.get('study_title', ''),
            record.get('study_abstract', ''),
            record.get('summary', ''),
            record.get('overall_design', ''),
            record.get('experiment_title', ''),
            record.get('gsm_description', '')
        ]
        
        combined_text = ' '.join([str(field).lower() for field in text_fields if field and str(field).lower() != 'null']).strip()
        
        # Enhanced RNA-seq analysis
        if combined_text and ('rna-seq' in combined_text or 'rnaseq' in combined_text or 'rna seq' in combined_text):
            
            # Direct single-cell indicators in the context
            for tech_type, keywords in sc_technologies.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        return True, f"RNA-seq with single-cell context: '{keyword}' found", 0.85, tech_type
            
            # Enhanced context analysis for ambiguous RNA-seq cases
            # Use AI to analyze if this is likely a single-cell experiment
            if self.ai_client and len(combined_text) > 50:
                ai_analysis = self._analyze_rna_seq_with_ai(combined_text, record)
                if ai_analysis['is_single_cell']:
                    return True, f"AI-identified single-cell RNA-seq: {ai_analysis['reason']}", ai_analysis['confidence'], 'scRNA-seq'
        
        # Additional check for common single-cell study patterns
        sc_study_patterns = [
            r'single.{0,5}cell.{0,10}(rna|transcriptom|gene|expression)',
            r'(10x|dropseq|smartseq|chromium).{0,20}(rna|transcriptom)',
            r'cell.{0,5}type.{0,10}(identification|classification|clustering)',
            r'cellular.{0,10}heterogeneity',
            r'(atlas|map).{0,10}single.{0,5}cell'
        ]
        
        for pattern in sc_study_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"Single-cell study pattern detected: {pattern}", 0.80, "scRNA-seq"
        
        return False, "No single-cell technology indicators found", 0.85, "unknown"
    
    def extract_sc_eqtl_criteria_optimized(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract 10 key sc-eQTL criteria specifically from sra_geo_ft2 data.
        
        Args:
            record: Database record
            
        Returns:
            Dict containing extracted criteria
        """
        criteria = {
            "organism": "",
            "tissue_type": "",
            "cell_type": "",
            "sample_size": "",
            "sequencing_platform": "",
            "project_id": "",
            "publication_info": "",
            "geographic_location": "",
            "age_range": "",
            "disease_status": ""
        }
        
        # 1. Organism (already handled by human detection)
        if record.get('organism_ch1'):
            criteria["organism"] = str(record['organism_ch1']).split(';')[0].strip()
        elif record.get('organism'):
            criteria["organism"] = str(record['organism']).strip()
        
        # 2. Extract tissue type
        tissue_sources = [
            record.get('tissue', ''),
            record.get('source_name', ''),
            record.get('characteristics_ch1', '')
        ]
        
        tissue_keywords = [
            'brain', 'liver', 'heart', 'lung', 'kidney', 'muscle', 'blood', 'skin',
            'bone', 'pancreas', 'breast', 'colon', 'stomach', 'intestine', 'spleen',
            'thymus', 'lymph', 'nerve', 'cortex', 'hippocampus', 'cerebellum'
        ]
        
        for source in tissue_sources:
            if source:
                source_lower = str(source).lower()
                for tissue in tissue_keywords:
                    if tissue in source_lower:
                        criteria["tissue_type"] = tissue.title()
                        break
                if criteria["tissue_type"]:
                    break
        
        # 3. Extract cell type information
        cell_info_sources = [
            record.get('characteristics_ch1', ''),
            record.get('sample_title', ''),
            record.get('study_title', '')
        ]
        
        cell_type_patterns = [
            r'cell\s*type\s*:\s*([^;,\n]+)',
            r'cells?\s*:\s*([^;,\n]+)',
            r'([a-z]+)\s*cells?',
            r'([a-z]+)\s*cell\s*type'
        ]
        
        for source in cell_info_sources:
            if source:
                source_text = str(source).lower()
                for pattern in cell_type_patterns:
                    match = re.search(pattern, source_text)
                    if match:
                        criteria["cell_type"] = match.group(1).strip().title()
                        break
                if criteria["cell_type"]:
                    break
        
        # 4. Extract sample size
        sample_size_sources = [
            record.get('study_title', ''),
            record.get('study_abstract', ''),
            record.get('spots', ''),
            record.get('bases', '')
        ]
        
        sample_patterns = [
            r'(\d+)\s*(?:samples?|subjects?|patients?|individuals?|donors?)',
            r'n\s*=\s*(\d+)',
            r'(\d+)\s*(?:single.?cells?|cells?)',
            r'from\s*(\d+)\s*(?:subjects?|patients?|individuals?)'
        ]
        
        for source in sample_size_sources[:2]:  # Only text sources
            if source:
                source_text = str(source).lower()
                for pattern in sample_patterns:
                    matches = re.findall(pattern, source_text)
                    if matches:
                        # Take the largest number found
                        max_size = max(int(match) for match in matches if match.isdigit())
                        criteria["sample_size"] = str(max_size)
                        break
                if criteria["sample_size"]:
                    break
        
        # Use spots/bases as fallback for sample size estimation
        if not criteria["sample_size"]:
            spots = record.get('spots', '')
            if spots and str(spots).isdigit():
                spot_count = int(spots)
                if spot_count > 1000000:  # >1M spots suggests substantial dataset
                    criteria["sample_size"] = f"~{spot_count//1000000}M spots"
        
        # 5. Sequencing platform
        platform_sources = [
            record.get('instrument_model', ''),
            record.get('platform', ''),
            record.get('study_title', '')
        ]
        
        for source in platform_sources:
            if source:
                source_text = str(source).strip()
                if source_text and source_text.lower() not in ['', 'null', 'none', 'na']:
                    criteria["sequencing_platform"] = source_text
                    break
        
        # 6. Project IDs (using actual table fields)
        project_ids = []
        id_fields = [
            ('sra_ID', record.get('sra_ID', '')),
            ('gsm_title', record.get('gsm_title', '')),
            ('gse_title', record.get('gse_title', '')),
            ('run_alias', record.get('run_alias', '')),
            ('experiment_alias', record.get('experiment_alias', '')),
            ('sample_alias', record.get('sample_alias', ''))
        ]
        
        for field_name, value in id_fields:
            if value and str(value).strip() and str(value).lower() != 'null':
                project_ids.append(f"{field_name}:{value}")
        
        criteria["project_id"] = "; ".join(project_ids)
        
        # 7. Publication info from pubmed_id
        pubmed_id = record.get('pubmed_id', '')
        if pubmed_id and str(pubmed_id).strip() and str(pubmed_id).lower() not in ['', 'null', 'none', 'na', '0']:
            criteria["publication_info"] = f"PMID: {pubmed_id}"
        else:
            criteria["publication_info"] = "Not published"
        
        # 8-10. Extract remaining criteria from study text using AI if available
        study_text = f"{record.get('study_title', '')} {record.get('study_abstract', '')}".strip()
        if study_text and self.ai_client:
            ai_extracted = self._extract_with_ai_partial(study_text)
            criteria.update(ai_extracted)
        else:
            # Set default values when AI is not available
            criteria["geographic_location"] = "Not specified"
            criteria["age_range"] = "Not specified"
            criteria["disease_status"] = "Not specified"
        
        return criteria
    
    def _analyze_rna_seq_with_ai(self, combined_text: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to analyze if an RNA-seq experiment is actually single-cell based on context.
        
        Args:
            combined_text: Combined text from multiple fields
            record: The database record
            
        Returns:
            Dict with analysis results
        """
        try:
            # Prepare context for AI analysis
            experiment_title = record.get('experiment_title', '')
            study_title = record.get('study_title', '')
            study_abstract = record.get('study_abstract', '')
            
            prompt = f"""
            Analyze this RNA-seq experiment to determine if it's actually a single-cell RNA-seq (scRNA-seq) experiment.

            Experiment Title: {experiment_title}
            Study Title: {study_title}
            Study Abstract: {study_abstract[:800]}

            Many single-cell experiments are labeled as just "RNA-Seq" in the experiment title, but the study context reveals they are actually single-cell experiments.

            Please analyze and respond with JSON format:
            {{
                "is_single_cell": true/false,
                "confidence": 0.0-1.0,
                "reason": "Brief explanation of your decision",
                "evidence": ["list", "of", "key", "evidence", "found"],
                "experiment_type": "scRNA-seq|scATAC-seq|bulk RNA-seq|unknown"
            }}

            Look for indicators like:
            - Single-cell technologies (10x, Smart-seq, Drop-seq, etc.)
            - Cell-level analysis mentions
            - Single-cell specific workflows
            - Cell type identification/clustering
            - Cellular heterogeneity studies
            - Individual cell mentions
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            response_text = response.content
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response_text)
                return {
                    "is_single_cell": analysis.get("is_single_cell", False),
                    "confidence": min(max(analysis.get("confidence", 0.5), 0.0), 1.0),
                    "reason": analysis.get("reason", "AI analysis completed"),
                    "evidence": analysis.get("evidence", []),
                    "experiment_type": analysis.get("experiment_type", "unknown")
                }
            except json.JSONDecodeError:
                # Fallback: analyze text response for key indicators
                response_lower = response_text.lower()
                is_sc = any(keyword in response_lower for keyword in [
                    "single cell", "single-cell", "scrna", "10x", "cell-level", "individual cells"
                ])
                return {
                    "is_single_cell": is_sc,
                    "confidence": 0.6 if is_sc else 0.3,
                    "reason": f"Text analysis: {response_text[:100]}...",
                    "evidence": [],
                    "experiment_type": "scRNA-seq" if is_sc else "unknown"
                }
                
        except Exception as e:
            logger.warning(f"AI RNA-seq analysis failed: {e}")
            return {
                "is_single_cell": False,
                "confidence": 0.0,
                "reason": f"AI analysis failed: {str(e)}",
                "evidence": [],
                "experiment_type": "unknown"
            }

    def _extract_with_ai_partial(self, study_text: str) -> Dict[str, str]:
        """
        Use AI to extract geographic, age, and disease information from study text.
        
        Args:
            study_text: Combined study title and abstract
            
        Returns:
            Dict with extracted information
        """
        try:
            prompt = f"""
            Analyze this scientific study description and extract the following information:

            Study Text: {study_text[:1000]}

            Please extract and return ONLY the following information in JSON format:
            {{
                "geographic_location": "Country or region if mentioned", 
                "age_range": "Age information if mentioned",
                "disease_status": "Disease, condition, or 'healthy' if mentioned"
            }}

            Use "Not specified" if information is not available.
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            response_text = response.content
            
            # Try to parse JSON response
            import json
            try:
                extracted = json.loads(response_text)
                return {
                    "geographic_location": extracted.get("geographic_location", "Not specified"),
                    "age_range": extracted.get("age_range", "Not specified"),
                    "disease_status": extracted.get("disease_status", "Not specified")
                }
            except json.JSONDecodeError:
                # Fallback to default values
                return {
                    "geographic_location": "Not specified",
                    "age_range": "Not specified", 
                    "disease_status": "Not specified"
                }
            
        except Exception as e:
            logger.warning(f"AI extraction failed: {e}")
            return {
                "geographic_location": "Not specified",
                "age_range": "Not specified",
                "disease_status": "Not specified"
            }
    
    def _extract_with_ai(self, study_text: str) -> Dict[str, str]:
        """
        Use AI to extract additional criteria from study text.
        
        Args:
            study_text: Combined study title and abstract
            
        Returns:
            Dict with extracted information
        """
        try:
            prompt = f"""
            Analyze this scientific study description and extract the following information:

            Study Text: {study_text[:1000]}

            Please extract and return ONLY the following information in JSON format:
            {{
                "publication_info": "PMID, DOI, or journal if mentioned",
                "geographic_location": "Country or region if mentioned", 
                "age_range": "Age information if mentioned",
                "disease_status": "Disease, condition, or 'healthy' if mentioned"
            }}

            Use "Not specified" if information is not available.
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            response_text = response.content
            
            # Try to parse JSON response
            import json
            try:
                extracted = json.loads(response_text)
                return {
                    "publication_info": extracted.get("publication_info", "Not specified"),
                    "geographic_location": extracted.get("geographic_location", "Not specified"),
                    "age_range": extracted.get("age_range", "Not specified"),
                    "disease_status": extracted.get("disease_status", "Not specified")
                }
            except json.JSONDecodeError:
                # Fallback to text parsing
                return {
                    "publication_info": "AI extraction failed",
                    "geographic_location": "AI extraction failed",
                    "age_range": "AI extraction failed", 
                    "disease_status": "AI extraction failed"
                }
            
        except Exception as e:
            logger.warning(f"AI extraction failed: {e}")
            return {
                "publication_info": "Not specified",
                "geographic_location": "Not specified",
                "age_range": "Not specified",
                "disease_status": "Not specified"
            }
    
    def filter_record_optimized(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply optimized filtering pipeline to a single record.
        
        Args:
            record: Database record from sra_geo_ft2
            
        Returns:
            Dict containing filtering results
        """
        # Generate record ID from available identifiers
        record_id = (record.get('sra_ID') or 
                    record.get('gsm_title') or 
                    record.get('run_alias') or 
                    record.get('experiment_alias') or 
                    f"row_{hash(str(record))%10000}")
        
        result = {
            "record_id": record_id,
            "passes_filter": False,  
            "filter_steps": {},
            "confidence_score": 0.0,
            "rejection_reason": "",
            "extracted_criteria": {},
            "processing_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Human sample check
            is_human, human_reason, human_confidence = self.is_human_sample_optimized(record)
            result["filter_steps"]["human_check"] = {
                "passed": is_human,
                "reason": human_reason,
                "confidence": human_confidence
            }
            
            if not is_human:
                result["rejection_reason"] = f"Not human sample: {human_reason}"
                result["processing_time"] = time.time() - start_time
                return result
            
            # Step 2: Cell line exclusion
            is_cell_line, cell_line_reason, cell_line_confidence = self.is_cell_line_sample(record)
            result["filter_steps"]["cell_line_check"] = {
                "passed": not is_cell_line,
                "reason": cell_line_reason,
                "confidence": cell_line_confidence
            }
            
            if is_cell_line:
                result["rejection_reason"] = f"Cell line sample: {cell_line_reason}"
                result["processing_time"] = time.time() - start_time
                return result
            
            # Step 3: Single-cell experiment check
            is_sc, sc_reason, sc_confidence, sc_type = self.is_single_cell_experiment_optimized(record)
            result["filter_steps"]["single_cell_check"] = {
                "passed": is_sc,
                "reason": sc_reason,
                "confidence": sc_confidence,
                "experiment_type": sc_type
            }
            
            if not is_sc:
                result["rejection_reason"] = f"Not single-cell experiment: {sc_reason}"
                result["processing_time"] = time.time() - start_time
                return result
            
            # Step 4: Extract sc-eQTL criteria
            criteria = self.extract_sc_eqtl_criteria_optimized(record)
            result["extracted_criteria"] = criteria
            
            # Calculate overall confidence score
            confidence_scores = [
                result["filter_steps"]["human_check"]["confidence"],
                result["filter_steps"]["cell_line_check"]["confidence"],
                result["filter_steps"]["single_cell_check"]["confidence"]
            ]
            result["confidence_score"] = np.mean(confidence_scores)
            
            # Record passes all filters
            result["passes_filter"] = True
            result["processing_time"] = time.time() - start_time
            
            # Update processing stats  
            self.processing_stats["total_processed"] += 1
            self.processing_stats["passed_filters"] += 1
            if sc_type != "unknown":
                self.processing_stats[f"sc_type_{sc_type}"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {record.get('run_accession', 'unknown')}: {e}")
            result["rejection_reason"] = f"Processing error: {str(e)}"
            result["processing_time"] = time.time() - start_time
            return result
    
    def batch_filter_optimized(self, 
                              batch_size: int = 5000,
                              max_records: Optional[int] = None,
                              output_file: Optional[str] = None,
                              enable_parallel: bool = True) -> Dict[str, Any]:
        """
        Batch filtering of sra_geo_ft2 table with optimized performance.
        
        Args:
            batch_size: Number of records per batch
            max_records: Maximum records to process (None for all)
            output_file: Output file path for results
            enable_parallel: Enable parallel processing
            
        Returns:
            Dict containing batch filtering results
        """
        logger.info(f"Starting batch filtering of {self.full_table_name}")
        
        start_time = time.time()
        results = []
        total_processed = 0
        
        self.processing_stats = defaultdict(int)
        
        try:
            conn = get_connection()
            
            # Get total record count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.full_table_name}")
                total_records = cur.fetchone()[0]
                logger.info(f"Total records in table: {total_records:,}")
            
            # Determine processing limit
            process_limit = min(max_records or total_records, total_records)
            
            # Process in batches
            offset = 0
            batch_num = 0
            
            while offset < process_limit:
                batch_num += 1
                current_batch_size = min(batch_size, process_limit - offset)
                
                # Fetch batch (using sra_ID for ordering since run_accession doesn't exist)
                query = f"""
                SELECT * FROM {self.full_table_name}
                ORDER BY "sra_ID"
                LIMIT %s OFFSET %s
                """
                
                with conn.cursor() as cur:
                    cur.execute(query, (current_batch_size, offset))
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    
                    batch_records = [dict(zip(columns, row)) for row in rows]
                
                logger.info(f"Processing batch {batch_num}: {len(batch_records)} records")
                
                # Process batch
                if enable_parallel and len(batch_records) > 100:
                    batch_results = self._process_batch_parallel(batch_records)
                else:
                    batch_results = [self.filter_record_optimized(record) for record in batch_records]
                
                # Filter passed records
                passed_results = [r for r in batch_results if r["passes_filter"]]
                results.extend(passed_results)
                
                total_processed += len(batch_records)
                
                # Progress logging
                pass_rate = (len(passed_results) / len(batch_records)) * 100 if batch_records else 0
                logger.info(f"Batch {batch_num} completed: {len(passed_results)}/{len(batch_records)} passed ({pass_rate:.1f}%)")
                
                offset += current_batch_size
                
                # Periodic progress report
                if batch_num % 10 == 0:
                    overall_pass_rate = (len(results) / total_processed) * 100
                    elapsed_time = time.time() - start_time
                    rate = total_processed / elapsed_time
                    logger.info(f"Progress: {total_processed:,}/{process_limit:,} processed "
                              f"({overall_pass_rate:.1f}% pass rate, {rate:.0f} records/sec)")
            
            conn.close()
            
            # Generate final report
            total_time = time.time() - start_time
            final_pass_rate = (len(results) / total_processed) * 100 if total_processed > 0 else 0
            
            report = {
                "processing_summary": {
                    "total_processed": total_processed,
                    "total_passed": len(results),
                    "pass_rate": final_pass_rate,
                    "processing_time": total_time,
                    "records_per_second": total_processed / total_time
                },
                "filter_statistics": dict(self.processing_stats),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results if requested
            if output_file:
                self._save_results(report, output_file)
            
            logger.info(f"Batch filtering completed: {len(results):,}/{total_processed:,} records passed "
                       f"({final_pass_rate:.1f}%) in {total_time:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Batch filtering failed: {e}")
            return {
                "processing_summary": {"status": "failed", "error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_batch_parallel(self, batch_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records in parallel."""
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_record = {
                executor.submit(self.filter_record_optimized, record): record 
                for record in batch_records
            }
            
            results = []
            for future in as_completed(future_to_record):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel processing error: {e}")
                    # Create error result
                    record = future_to_record[future]
                    # Generate record ID using available fields
                    record_id = (record.get('sra_ID') or 
                                record.get('gsm_title') or 
                                record.get('run_alias') or 
                                f"error_{hash(str(record))%10000}")
                    error_result = {
                        "record_id": record_id,
                        "passes_filter": False,
                        "rejection_reason": f"Processing error: {str(e)}",
                        "confidence_score": 0.0
                    }
                    results.append(error_result)
            
            return results
    
    def _save_results(self, report: Dict[str, Any], output_file: str):
        """Save filtering results to file."""
        
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            elif output_file.endswith('.csv'):
                import pandas as pd
                
                # Flatten results for CSV
                flat_results = []
                for result in report['results']:
                    flat_record = {
                        'record_id': result['record_id'],
                        'passes_filter': result['passes_filter'],
                        'confidence_score': result['confidence_score'],
                        'processing_time': result['processing_time']
                    }
                    
                    # Add filter step results
                    for step, step_result in result.get('filter_steps', {}).items():
                        flat_record[f'{step}_passed'] = step_result.get('passed', False)
                        flat_record[f'{step}_confidence'] = step_result.get('confidence', 0.0)
                    
                    # Add extracted criteria
                    for key, value in result.get('extracted_criteria', {}).items():
                        flat_record[f'criteria_{key}'] = value
                    
                    flat_results.append(flat_record)
                
                df = pd.DataFrame(flat_results)
                df.to_csv(output_file, index=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
    
    def get_table_preview(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get a preview of the sra_geo_ft2 table structure and sample data.
        
        Args:
            limit: Number of sample records to return
            
        Returns:
            Dict containing table preview information
        """
        try:
            conn = get_connection()
            
            # Get table structure
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (self.schema, self.table))
                
                columns = [{"name": row[0], "type": row[1], "nullable": row[2]} for row in cur.fetchall()]
            
            # Get sample data
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.full_table_name} LIMIT %s", (limit,))
                column_names = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                
                sample_data = [dict(zip(column_names, row)) for row in rows]
            
            # Get record count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.full_table_name}")
                total_records = cur.fetchone()[0]
            
            conn.close()
            
            return {
                "table_info": {
                    "schema": self.schema,
                    "table": self.table, 
                    "total_columns": len(columns),
                    "total_records": total_records
                },
                "columns": columns,
                "sample_data": sample_data,
                "relevant_columns": {
                    "human_identification": ["organism_ch1", "organism", "scientific_name", "experiment_title"],
                    "cell_line_detection": ["characteristics_ch1", "source_name_ch1", "gsm_description"],
                    "single_cell_detection": ["study_title", "study_abstract", "summary", "overall_design", "technology"],
                    "criteria_extraction": ["study_title", "study_abstract", "summary", "spots", "bases", "pubmed_id"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get table preview: {e}")
            return {"error": str(e)}


def test_sra_geo_ft2_optimizer():
    """Test function for the sra_geo_ft2 optimizer."""
    
    logger.info("Testing SRA-GEO-FT2 Optimizer...")
    
    try:
        # Initialize optimizer
        optimizer = SraGeoFt2Optimizer()
        
        # Get table preview
        print("1. Getting table preview...")
        preview = optimizer.get_table_preview(limit=5)
        
        if "error" not in preview:
            print(f"✓ Table: {preview['table_info']['total_records']:,} records, {preview['table_info']['total_columns']} columns")
            print(f"✓ Relevant columns identified: {list(preview['relevant_columns'].keys())}")
        else:
            print(f"✗ Preview failed: {preview['error']}")
            return False
        
        # Test small batch filtering
        print("2. Testing small batch filtering...")
        test_results = optimizer.batch_filter_optimized(
            batch_size=1000,
            max_records=5000,
            enable_parallel=False
        )
        
        if test_results["processing_summary"].get("status") != "failed":
            summary = test_results["processing_summary"]
            print(f"✓ Processed {summary['total_processed']:,} records")
            print(f"✓ {summary['total_passed']:,} records passed ({summary['pass_rate']:.1f}%)")
            print(f"✓ Processing rate: {summary['records_per_second']:.0f} records/sec")
        else:
            print(f"✗ Batch filtering failed: {test_results['processing_summary']['error']}")
            return False
        
        # Show filter statistics
        if test_results.get("filter_statistics"):
            print("✓ Filter statistics:", dict(test_results["filter_statistics"]))
        
        print("✓ SRA-GEO-FT2 optimizer test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    test_sra_geo_ft2_optimizer() 