"""
Enhanced database handler for merged sra_geo_ft2 table with advanced optimization strategies.
Supports millions of records with intelligent filtering and relationship optimization.
"""

import psycopg2
import psycopg2.extras
from typing import Dict, List, Any, Optional, Tuple, Set, Generator
import logging
import pandas as pd
from .connect import get_connection, get_cursor
from datetime import datetime
import json
import re
from collections import defaultdict, Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class MergedTableHandler:
    """
    Advanced handler for the merged sra_geo_ft2 table with optimization strategies.
    """
    
    def __init__(self, schema: str = "scagent", table: str = "merged/sra_geo_ft2"):
        self.schema = schema
        self.table = table
        self.full_table_name = f'"{schema}"."{table}"'
        self._column_cache = None
        self._column_mapping = None
        self._relationship_cache = None
        
    def discover_table_structure(self, conn: Optional[psycopg2.extensions.connection] = None) -> Dict[str, Any]:
        """
        Comprehensive table structure discovery including column analysis and relationship mapping.
        
        Returns:
            Dict containing complete table metadata
        """
        should_close = False
        if conn is None:
            conn = get_connection()
            should_close = True
            
        try:
            with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
                # Get basic table info
                cur.execute(f"""
                    SELECT 
                        schemaname, 
                        tablename, 
                        tableowner,
                        tablespace,
                        hasindexes,
                        hasrules,
                        hastriggers
                    FROM pg_tables 
                    WHERE schemaname = %s AND tablename = %s
                """, (self.schema, self.table.split('/')[-1]))
                
                table_info = cur.fetchone()
                if not table_info:
                    raise ValueError(f"Table {self.full_table_name} not found")
                
                # Get column information with detailed statistics
                cur.execute(f"""
                    SELECT 
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default,
                        c.character_maximum_length,
                        c.numeric_precision,
                        c.numeric_scale,
                        c.ordinal_position
                    FROM information_schema.columns c
                    WHERE c.table_schema = %s AND c.table_name = %s
                    ORDER BY c.ordinal_position
                """, (self.schema, self.table.split('/')[-1]))
                
                columns = cur.fetchall()
                
                # Get table size and row count
                cur.execute(f"""
                    SELECT 
                        COUNT(*) as row_count,
                        pg_size_pretty(pg_total_relation_size('{self.full_table_name}')) as table_size,
                        pg_total_relation_size('{self.full_table_name}') as table_size_bytes
                """)
                
                size_info = cur.fetchone()
                
                # Analyze column content and null distribution
                column_analysis = self._analyze_column_content(cur, [col['column_name'] for col in columns])
                
                structure = {
                    "table_info": dict(table_info),
                    "basic_stats": dict(size_info),
                    "columns": [dict(col) for col in columns],
                    "column_analysis": column_analysis,
                    "discovery_timestamp": datetime.now().isoformat(),
                    "total_columns": len(columns)
                }
                
                # Cache the results
                self._column_cache = structure
                
                return structure
                
        finally:
            if should_close:
                conn.close()
    
    def _analyze_column_content(self, cur, column_names: List[str]) -> Dict[str, Any]:
        """
        Analyze column content distribution and identify relevant fields for sc-eQTL analysis.
        """
        analysis = {}
        # 强制采样不超过1000行
        sample_size = 1000
        for col_name in column_names:
            try:
                # Get null percentage and unique value count (采样)
                cur.execute(f'''
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT("{col_name}") as non_null_count,
                        COUNT(DISTINCT "{col_name}") as unique_count,
                        ROUND(
                            (COUNT(*) - COUNT("{col_name}"))::numeric / COUNT(*) * 100, 2
                        ) as null_percentage
                    FROM (SELECT "{col_name}" FROM {self.full_table_name} LIMIT %s) AS sample
                ''', (sample_size,))
                stats = cur.fetchone()
                # Get sample values for content analysis (采样)
                cur.execute(f'''
                    SELECT "{col_name}", COUNT(*) as freq
                    FROM (SELECT "{col_name}" FROM {self.full_table_name} WHERE "{col_name}" IS NOT NULL LIMIT %s) AS sample
                    GROUP BY "{col_name}"
                    ORDER BY freq DESC
                    LIMIT 20
                ''', (sample_size,))
                sample_values = cur.fetchall()
                # Identify potential field types based on name and content
                field_type = self._classify_field_type(col_name, sample_values)
                analysis[col_name] = {
                    "total_count": stats["total_count"],
                    "non_null_count": stats["non_null_count"], 
                    "unique_count": stats["unique_count"],
                    "null_percentage": float(stats["null_percentage"]),
                    "sample_values": [dict(v) for v in sample_values],
                    "field_type": field_type,
                    "sc_eqtl_relevance": self._assess_sc_eqtl_relevance(col_name, field_type, sample_values)
                }
            except Exception as e:
                logger.warning(f"Failed to analyze column {col_name}: {e}")
                analysis[col_name] = {
                    "error": str(e),
                    "field_type": "unknown",
                    "sc_eqtl_relevance": 0
                }
        return analysis
    
    def _classify_field_type(self, col_name: str, sample_values: List[Dict]) -> str:
        """
        Classify field type based on column name and sample values.
        
        Args:
            col_name: Column name
            sample_values: Sample values from the column
            
        Returns:
            Field type classification
        """
        col_lower = col_name.lower()
        
        # ID fields
        if any(keyword in col_lower for keyword in ['accession', 'id', 'identifier']):
            if 'study' in col_lower:
                return 'study_identifier'
            elif 'run' in col_lower:
                return 'run_identifier'
            elif 'sample' in col_lower:
                return 'sample_identifier'
            elif 'experiment' in col_lower:
                return 'experiment_identifier'
            else:
                return 'generic_identifier'
        
        # Organism/species fields
        elif any(keyword in col_lower for keyword in ['organism', 'species', 'scientific_name', 'taxon']):
            return 'organism'
        
        # Tissue/cell type fields
        elif any(keyword in col_lower for keyword in ['tissue', 'cell_type', 'cell_line', 'organ']):
            return 'tissue_cell'
            
        # Sequencing method fields
        elif any(keyword in col_lower for keyword in ['platform', 'instrument', 'library', 'sequencing']):
            return 'sequencing_method'
            
        # Metadata fields
        elif any(keyword in col_lower for keyword in ['title', 'description', 'summary', 'abstract']):
            return 'metadata_text'
            
        # Date fields
        elif any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated']):
            return 'date_time'
            
        # Count/numeric fields
        elif any(keyword in col_lower for keyword in ['count', 'number', 'size', 'length']):
            return 'numeric_count'
            
        # Geographic fields
        elif any(keyword in col_lower for keyword in ['country', 'location', 'geographic']):
            return 'geographic'
            
        # Publication fields
        elif any(keyword in col_lower for keyword in ['publication', 'pubmed', 'doi', 'author']):
            return 'publication'
            
        else:
            return 'other'
    
    def _assess_sc_eqtl_relevance(self, col_name: str, field_type: str, sample_values: List[Dict]) -> int:
        """
        Assess column relevance for sc-eQTL analysis (0-5 scale).
        
        Args:
            col_name: Column name
            field_type: Classified field type
            sample_values: Sample values from the column
            
        Returns:
            Relevance score (0-5)
        """
        # High relevance fields (required for sc-eQTL)
        if field_type in ['organism', 'tissue_cell', 'study_identifier', 'run_identifier']:
            return 5
        
        # Medium-high relevance
        elif field_type in ['sequencing_method', 'sample_identifier']:
            return 4
            
        # Medium relevance
        elif field_type in ['metadata_text', 'publication']:
            return 3
            
        # Low-medium relevance
        elif field_type in ['numeric_count', 'date_time']:
            return 2
            
        # Low relevance
        elif field_type in ['geographic']:
            return 1
            
        else:
            return 0
    
    def create_column_mapping(self, conn: Optional[psycopg2.extensions.connection] = None) -> Dict[str, Any]:
        """
        Create intelligent column mapping for 6 required + 4 optional conditions.
        
        Required fields:
        1. organism/species
        2. tissue/cell_type
        3. study_accession
        4. run_accession
        5. sequencing_method
        6. sample_size/count
        
        Optional fields:
        1. publication_info
        2. geographic_location
        3. age_info
        4. disease_annotation
        
        Returns:
            Dict containing field mappings and confidence scores
        """
        if self._column_cache is None:
            self.discover_table_structure(conn)
        
        mapping = {
            "required_fields": {},
            "optional_fields": {},
            "unmapped_columns": [],
            "mapping_confidence": {},
            "field_combinations": {}
        }
        
        column_analysis = self._column_cache["column_analysis"]
        
        # Map required fields
        required_mapping = {
            "organism": self._find_best_columns(column_analysis, ["organism", "species", "scientific_name", "taxon"]),
            "tissue_cell": self._find_best_columns(column_analysis, ["tissue", "cell_type", "cell_line", "organ"]),
            "study_accession": self._find_best_columns(column_analysis, ["study_accession", "study_id"]),
            "run_accession": self._find_best_columns(column_analysis, ["run_accession", "run_id"]),
            "sequencing_method": self._find_best_columns(column_analysis, ["platform", "instrument", "library_strategy"]),
            "sample_size": self._find_best_columns(column_analysis, ["sample_count", "spots", "reads", "cells"])
        }
        
        # Map optional fields
        optional_mapping = {
            "publication": self._find_best_columns(column_analysis, ["pubmed", "doi", "publication", "pmid"]),
            "geographic": self._find_best_columns(column_analysis, ["country", "location", "geographic"]),
            "age": self._find_best_columns(column_analysis, ["age", "developmental_stage"]),
            "disease": self._find_best_columns(column_analysis, ["disease", "tumor", "cancer", "condition"])
        }
        
        mapping["required_fields"] = required_mapping
        mapping["optional_fields"] = optional_mapping
        
        # Calculate mapping confidence
        for field_type, candidates in {**required_mapping, **optional_mapping}.items():
            if candidates:
                best_candidate = candidates[0]
                confidence = self._calculate_mapping_confidence(best_candidate, field_type)
                mapping["mapping_confidence"][field_type] = confidence
        
        # Identify unmapped columns
        mapped_columns = set()
        for candidates in {**required_mapping, **optional_mapping}.values():
            mapped_columns.update([c["column_name"] for c in candidates])
        
        all_columns = set(column_analysis.keys())
        mapping["unmapped_columns"] = list(all_columns - mapped_columns)
        
        # Cache the mapping
        self._column_mapping = mapping
        
        return mapping
    
    def _find_best_columns(self, column_analysis: Dict, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find best matching columns for given keywords.
        
        Args:
            column_analysis: Column analysis results
            keywords: Keywords to search for
            
        Returns:
            List of matching columns sorted by relevance
        """
        candidates = []
        
        for col_name, analysis in column_analysis.items():
            score = 0
            col_lower = col_name.lower()
            
            # Exact keyword match
            for keyword in keywords:
                if keyword.lower() in col_lower:
                    score += 10
            
            # Partial keyword match
            for keyword in keywords:
                if any(part in col_lower for part in keyword.lower().split('_')):
                    score += 5
            
            # Boost score based on data quality
            null_percentage = analysis.get("null_percentage", 100)
            if null_percentage < 50:
                score += 3
            elif null_percentage < 80:
                score += 1
            
            # Boost score based on unique count
            unique_count = analysis.get("unique_count", 0)
            total_count = analysis.get("total_count", 1)
            if unique_count > 1 and unique_count < total_count * 0.8:  # Good diversity
                score += 2
            
            if score > 0:
                candidates.append({
                    "column_name": col_name,
                    "score": score,
                    "null_percentage": null_percentage,
                    "unique_count": unique_count,
                    "analysis": analysis
                })
        
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates
    
    def _calculate_mapping_confidence(self, candidate: Dict, field_type: str) -> float:
        """
        Calculate confidence score for a field mapping.
        
        Args:
            candidate: Candidate column information
            field_type: Target field type
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.0
        
        # Base confidence from matching score
        confidence += min(candidate["score"] / 20.0, 0.5)
        
        # Data quality contribution
        null_percentage = candidate["null_percentage"]
        if null_percentage < 10:
            confidence += 0.3
        elif null_percentage < 30:
            confidence += 0.2
        elif null_percentage < 50:
            confidence += 0.1
        
        # Unique value distribution
        unique_count = candidate["unique_count"]
        total_count = candidate["analysis"]["total_count"]
        
        if total_count > 0:
            diversity = unique_count / total_count
            if field_type in ["organism", "tissue_cell"] and 0.1 <= diversity <= 0.5:
                confidence += 0.2
            elif field_type in ["study_accession", "run_accession"] and diversity > 0.8:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def optimize_study_run_relationships(self, conn: Optional[psycopg2.extensions.connection] = None) -> Dict[str, Any]:
        """
        Analyze and optimize sra_study_accession to sra_run_accession one-to-many relationships.
        
        Returns:
            Dict containing relationship analysis and optimization strategies
        """
        should_close = False
        if conn is None:
            conn = get_connection()
            should_close = True
        
        try:
            if self._column_mapping is None:
                self.create_column_mapping(conn)
            
            study_cols = self._column_mapping["required_fields"]["study_accession"]
            run_cols = self._column_mapping["required_fields"]["run_accession"]
            
            if not study_cols or not run_cols:
                return {"error": "Could not identify study and run accession columns"}
            
            study_col = study_cols[0]["column_name"]
            run_col = run_cols[0]["column_name"]
            
            with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
                # Analyze relationship cardinality
                cur.execute(f"""
                    SELECT 
                        "{study_col}" as study_acc,
                        COUNT("{run_col}") as run_count,
                        ARRAY_AGG("{run_col}") as run_accessions
                    FROM {self.full_table_name}
                    WHERE "{study_col}" IS NOT NULL AND "{run_col}" IS NOT NULL
                    GROUP BY "{study_col}"
                    ORDER BY run_count DESC
                    LIMIT 1000
                """)
                
                relationships = cur.fetchall()
                
                # Calculate statistics
                run_counts = [r["run_count"] for r in relationships]
                relationship_stats = {
                    "total_studies": len(relationships),
                    "total_runs": sum(run_counts),
                    "avg_runs_per_study": np.mean(run_counts) if run_counts else 0,
                    "median_runs_per_study": np.median(run_counts) if run_counts else 0,
                    "max_runs_per_study": max(run_counts) if run_counts else 0,
                    "min_runs_per_study": min(run_counts) if run_counts else 0,
                    "studies_with_multiple_runs": sum(1 for c in run_counts if c > 1),
                    "multi_run_percentage": (sum(1 for c in run_counts if c > 1) / len(run_counts) * 100) if run_counts else 0
                }
                
                # Optimization strategies
                optimization_strategies = self._generate_relationship_optimization_strategies(relationship_stats, relationships)
                
                return {
                    "study_column": study_col,
                    "run_column": run_col,
                    "relationship_stats": relationship_stats,
                    "sample_relationships": [dict(r) for r in relationships[:10]],
                    "optimization_strategies": optimization_strategies,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
        finally:
            if should_close:
                conn.close()
    
    def _generate_relationship_optimization_strategies(self, stats: Dict, relationships: List[Dict]) -> Dict[str, Any]:
        """
        Generate optimization strategies for study-run relationships.
        
        Args:
            stats: Relationship statistics
            relationships: Sample relationship data
            
        Returns:
            Dict containing optimization strategies
        """
        strategies = {
            "display_optimization": {},
            "query_optimization": {},
            "filtering_optimization": {}
        }
        
        # Display optimization
        if stats["multi_run_percentage"] > 50:
            strategies["display_optimization"] = {
                "recommendation": "use_hierarchical_display",
                "description": "Use hierarchical display with study as parent and runs as children",
                "benefits": ["Reduced visual clutter", "Better data organization", "Faster user comprehension"],
                "implementation": {
                    "group_by_study": True,
                    "show_run_count": True,
                    "collapsible_runs": True,
                    "summary_statistics": ["total_runs", "avg_file_size", "sequencing_platforms"]
                }
            }
        
        # Query optimization
        if stats["avg_runs_per_study"] > 10:
            strategies["query_optimization"] = {
                "recommendation": "use_study_level_filtering",
                "description": "Filter at study level first, then expand to runs",
                "benefits": ["Reduced query complexity", "Faster initial results", "Better memory usage"],
                "implementation": {
                    "two_stage_query": True,
                    "study_level_cache": True,
                    "lazy_run_loading": True,
                    "pagination_by_study": True
                }
            }
        
        # Filtering optimization
        high_diversity_studies = [r for r in relationships if r["run_count"] > stats["median_runs_per_study"] * 2]
        if len(high_diversity_studies) > stats["total_studies"] * 0.2:
            strategies["filtering_optimization"] = {
                "recommendation": "study_diversity_weighting",
                "description": "Weight study importance by run diversity and quality",
                "benefits": ["Better quality results", "Reduced noise", "Focus on comprehensive studies"],
                "implementation": {
                    "diversity_scoring": True,
                    "quality_weighting": True,
                    "comprehensive_study_boost": True,
                    "min_runs_threshold": max(3, int(stats["median_runs_per_study"]))
                }
            }
        
        return strategies
    
    def full_scan_query(self, 
                       conditions: Optional[Dict[str, Any]] = None,
                       batch_size: int = 10000,
                       max_records: Optional[int] = None,
                       enable_parallel: bool = True,
                       conn: Optional[psycopg2.extensions.connection] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Full scan query with intelligent batching for millions of records.
        
        Args:
            conditions: Filter conditions
            batch_size: Records per batch
            max_records: Maximum records to process (None for unlimited)
            enable_parallel: Enable parallel processing
            conn: Database connection
            
        Yields:
            Batches of records
        """
        should_close = False
        if conn is None:
            conn = get_connection()
            should_close = True
        
        try:
            # Get total count first
            count_query = f"SELECT COUNT(*) FROM {self.full_table_name}"
            count_params = []
            
            if conditions:
                where_clause, count_params = self._build_where_clause(conditions)
                count_query += f" WHERE {where_clause}"
            
            with get_cursor(conn) as cur:
                cur.execute(count_query, count_params)
                total_records = cur.fetchone()[0]
                
                logger.info(f"Full scan: {total_records} total records found")
                
                if max_records:
                    total_records = min(total_records, max_records)
                
                # Calculate optimal batch size for large datasets
                optimal_batch_size = self._calculate_optimal_batch_size(total_records, batch_size)
                
                # Generate batches
                processed_records = 0
                offset = 0
                
                while processed_records < total_records:
                    current_batch_size = min(optimal_batch_size, total_records - processed_records)
                    
                    # Build query
                    query = f"SELECT * FROM {self.full_table_name}"
                    params = []
                    
                    if conditions:
                        where_clause, params = self._build_where_clause(conditions)
                        query += f" WHERE {where_clause}"
                    
                    query += f" LIMIT %s OFFSET %s"
                    params.extend([current_batch_size, offset])
                    
                    # Execute query
                    start_time = time.time()
                    
                    with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(query, params)
                        batch_results = [dict(row) for row in cur.fetchall()]
                    
                    query_time = time.time() - start_time
                    
                    if not batch_results:
                        break
                    
                    logger.info(f"Batch {offset//optimal_batch_size + 1}: {len(batch_results)} records in {query_time:.2f}s")
                    
                    yield batch_results
                    
                    processed_records += len(batch_results)
                    offset += optimal_batch_size
                    
                    # Progress logging
                    if processed_records % (optimal_batch_size * 10) == 0:
                        progress = (processed_records / total_records) * 100
                        logger.info(f"Full scan progress: {processed_records}/{total_records} ({progress:.1f}%)")
                
        finally:
            if should_close:
                conn.close()
    
    def _build_where_clause(self, conditions: Dict[str, Any]) -> Tuple[str, List]:
        """
        Build WHERE clause from conditions dictionary.
        
        Args:
            conditions: Filter conditions
            
        Returns:
            Tuple of (where_clause, parameters)
        """
        clauses = []
        params = []
        
        for column, value in conditions.items():
            if isinstance(value, list):
                # IN clause
                placeholders = ','.join(['%s'] * len(value))
                clauses.append(f'"{column}" IN ({placeholders})')
                params.extend(value)
            elif isinstance(value, dict):
                if 'like' in value:
                    clauses.append(f'"{column}" ILIKE %s')
                    params.append(f"%{value['like']}%")
                elif 'range' in value:
                    if 'min' in value['range']:
                        clauses.append(f'"{column}" >= %s')
                        params.append(value['range']['min'])
                    if 'max' in value['range']:
                        clauses.append(f'"{column}" <= %s')
                        params.append(value['range']['max'])
                elif 'null' in value:
                    if value['null']:
                        clauses.append(f'"{column}" IS NULL')
                    else:
                        clauses.append(f'"{column}" IS NOT NULL')
            else:
                # Equality
                clauses.append(f'"{column}" = %s')
                params.append(value)
        
        return ' AND '.join(clauses), params
    
    def _calculate_optimal_batch_size(self, total_records: int, requested_batch_size: int) -> int:
        """
        Calculate optimal batch size based on total records and system constraints.
        
        Args:
            total_records: Total number of records
            requested_batch_size: Requested batch size
            
        Returns:
            Optimal batch size
        """
        # For very large datasets, use larger batches to reduce overhead
        if total_records > 1000000:  # 1M+ records
            return max(requested_batch_size, 50000)
        elif total_records > 100000:  # 100K+ records
            return max(requested_batch_size, 10000)
        else:
            return requested_batch_size
    
    def dynamic_field_concatenation(self, 
                                  record: Dict[str, Any], 
                                  field_mapping: Dict[str, List[str]],
                                  use_ai_fallback: bool = True) -> Dict[str, Any]:
        """
        Perform dynamic field concatenation for NULL values using intelligent strategies.
        
        Args:
            record: Single record to process
            field_mapping: Mapping of logical fields to column names
            use_ai_fallback: Whether to use AI for complex field inference
            
        Returns:
            Record with concatenated/inferred fields
        """
        enhanced_record = record.copy()
        
        for logical_field, column_candidates in field_mapping.items():
            # Try to find non-null value from candidate columns
            final_value = None
            source_columns = []
            
            for col_name in column_candidates:
                if col_name in record and record[col_name] is not None:
                    value = str(record[col_name]).strip()
                    if value:  # Non-empty string
                        if final_value is None:
                            final_value = value
                            source_columns.append(col_name)
                        else:
                            # Concatenate different values
                            if value.lower() != final_value.lower():
                                final_value = f"{final_value}; {value}"
                                source_columns.append(col_name)
            
            # If no direct value found, try intelligent inference
            if final_value is None and use_ai_fallback:
                final_value = self._infer_field_value(record, logical_field)
                if final_value:
                    source_columns = ["AI_INFERRED"]
            
            # Store result
            enhanced_record[f"enhanced_{logical_field}"] = {
                "value": final_value,
                "source_columns": source_columns,
                "inference_method": "concatenation" if source_columns and "AI_INFERRED" not in source_columns else "ai_inference"
            }
        
        return enhanced_record
    
    def _infer_field_value(self, record: Dict[str, Any], logical_field: str) -> Optional[str]:
        """
        Use intelligent strategies to infer field values from available data.
        
        Args:
            record: Record data
            logical_field: Logical field to infer
            
        Returns:
            Inferred value or None
        """
        # Text fields that might contain the information
        text_fields = []
        for key, value in record.items():
            if isinstance(value, str) and len(value) > 10:  # Potentially informative text
                text_fields.append(value)
        
        if not text_fields:
            return None
        
        # Field-specific inference strategies
        if logical_field == "organism":
            return self._infer_organism_from_text(text_fields)
        elif logical_field == "tissue_cell":
            return self._infer_tissue_from_text(text_fields)
        elif logical_field == "sequencing_method":
            return self._infer_sequencing_method_from_text(text_fields)
        else:
            return None
    
    def _infer_organism_from_text(self, text_fields: List[str]) -> Optional[str]:
        """Infer organism from text content."""
        common_organisms = [
            "Homo sapiens", "human", "Mus musculus", "mouse", "Rattus norvegicus", "rat",
            "Drosophila melanogaster", "fly", "Caenorhabditis elegans", "worm",
            "Danio rerio", "zebrafish", "Arabidopsis thaliana", "Saccharomyces cerevisiae"
        ]
        
        text_combined = " ".join(text_fields).lower()
        
        for organism in common_organisms:
            if organism.lower() in text_combined:
                return organism
        
        return None
    
    def _infer_tissue_from_text(self, text_fields: List[str]) -> Optional[str]:
        """Infer tissue/cell type from text content."""
        common_tissues = [
            "brain", "liver", "kidney", "heart", "lung", "muscle", "blood", "bone marrow",
            "skin", "pancreas", "thymus", "spleen", "lymph node", "breast", "prostate",
            "ovary", "testis", "embryo", "stem cell", "T cell", "B cell", "neuron",
            "fibroblast", "epithelial", "endothelial", "macrophage", "monocyte"
        ]
        
        text_combined = " ".join(text_fields).lower()
        
        for tissue in common_tissues:
            if tissue.lower() in text_combined:
                return tissue
        
        return None
    
    def _infer_sequencing_method_from_text(self, text_fields: List[str]) -> Optional[str]:
        """Infer sequencing method from text content."""
        sequencing_methods = [
            "Illumina", "HiSeq", "NextSeq", "NovaSeq", "MiSeq", "Ion Torrent", "PacBio",
            "Oxford Nanopore", "10x Genomics", "Smart-seq", "Drop-seq", "inDrop",
            "CEL-seq", "MARS-seq", "Fluidigm", "single cell", "scRNA-seq", "RNA-seq"
        ]
        
        text_combined = " ".join(text_fields).lower()
        
        for method in sequencing_methods:
            if method.lower() in text_combined:
                return method
        
        return None
    
    def get_table_summary(self, conn: Optional[psycopg2.extensions.connection] = None) -> Dict[str, Any]:
        """
        Get comprehensive table summary for the merged table.
        
        Returns:
            Dict containing table summary information
        """
        if self._column_cache is None:
            structure = self.discover_table_structure(conn)
        else:
            structure = self._column_cache
        
        if self._column_mapping is None:
            mapping = self.create_column_mapping(conn)
        else:
            mapping = self._column_mapping
        
        # Calculate summary statistics
        total_columns = structure["total_columns"]
        mapped_columns = len([c for candidates in mapping["required_fields"].values() for c in candidates])
        mapped_columns += len([c for candidates in mapping["optional_fields"].values() for c in candidates])
        
        high_quality_columns = len([
            col for col, analysis in structure["column_analysis"].items()
            if analysis.get("null_percentage", 100) < 20
        ])
        
        sc_eqtl_relevant_columns = len([
            col for col, analysis in structure["column_analysis"].items()
            if analysis.get("sc_eqtl_relevance", 0) >= 3
        ])
        
        summary = {
            "table_name": self.full_table_name,
            "basic_statistics": {
                "total_records": structure["basic_stats"]["row_count"],
                "total_columns": total_columns,
                "table_size": structure["basic_stats"]["table_size"],
                "table_size_bytes": structure["basic_stats"]["table_size_bytes"]
            },
            "data_quality": {
                "high_quality_columns": high_quality_columns,
                "high_quality_percentage": (high_quality_columns / total_columns * 100) if total_columns > 0 else 0,
                "mapped_columns": mapped_columns,
                "mapping_coverage": (mapped_columns / total_columns * 100) if total_columns > 0 else 0,
                "sc_eqtl_relevant_columns": sc_eqtl_relevant_columns
            },
            "field_mapping": {
                "required_fields_mapped": len([f for f, candidates in mapping["required_fields"].items() if candidates]),
                "optional_fields_mapped": len([f for f, candidates in mapping["optional_fields"].items() if candidates]),
                "total_required_fields": 6,
                "total_optional_fields": 4
            },
            "recommendations": self._generate_processing_recommendations(structure, mapping),
            "summary_timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def _generate_processing_recommendations(self, structure: Dict, mapping: Dict) -> List[Dict[str, str]]:
        """Generate processing recommendations based on table analysis."""
        recommendations = []
        
        total_records = structure["basic_stats"]["row_count"]
        
        # Batch size recommendations
        if total_records > 1000000:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "recommendation": "Use large batch sizes (50K-100K) for full scans",
                "reason": f"Table has {total_records:,} records"
            })
        
        # Column mapping recommendations
        unmapped_required = len([f for f, candidates in mapping["required_fields"].items() if not candidates])
        if unmapped_required > 0:
            recommendations.append({
                "type": "data_quality",
                "priority": "critical",
                "recommendation": f"Manual column mapping needed for {unmapped_required} required fields",
                "reason": "Some required fields could not be automatically mapped"
            })
        
        # Null handling recommendations
        high_null_columns = len([
            col for col, analysis in structure["column_analysis"].items()
            if analysis.get("null_percentage", 0) > 50
        ])
        
        if high_null_columns > structure["total_columns"] * 0.3:
            recommendations.append({
                "type": "data_processing",
                "priority": "medium",
                "recommendation": "Enable dynamic field concatenation for NULL handling",
                "reason": f"{high_null_columns} columns have >50% NULL values"
            })
        
        return recommendations

def test_merged_table_handler():
    """Test function for the merged table handler."""
    handler = MergedTableHandler()
    
    try:
        # Test structure discovery
        print("Testing table structure discovery...")
        structure = handler.discover_table_structure()
        print(f"✓ Discovered {structure['total_columns']} columns")
        
        # Test column mapping
        print("Testing column mapping...")
        mapping = handler.create_column_mapping()
        required_mapped = len([f for f, candidates in mapping["required_fields"].items() if candidates])
        print(f"✓ Mapped {required_mapped}/6 required fields")
        
        # Test relationship optimization
        print("Testing relationship optimization...")
        relationships = handler.optimize_study_run_relationships()
        if "error" not in relationships:
            print(f"✓ Analyzed {relationships['relationship_stats']['total_studies']} study-run relationships")
        
        # Test summary
        print("Testing table summary...")
        summary = handler.get_table_summary()
        print(f"✓ Generated summary for {summary['basic_statistics']['total_records']:,} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_merged_table_handler() 