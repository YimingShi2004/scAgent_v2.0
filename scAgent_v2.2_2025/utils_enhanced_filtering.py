"""
Enhanced filtering system for scAgent with full-scan capabilities and intelligent content recognition.
Integrates with MergedTableHandler for processing millions of records efficiently.
"""

import logging
from typing import Dict, List, Any, Optional, Generator, Tuple
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict, Counter

from .db.merged_table_handler import MergedTableHandler
from scAgent.models.client import get_model_client
from .utils import apply_intelligent_sc_eqtl_filters

logger = logging.getLogger(__name__)

class EnhancedScEQTLFilter:
    """
    Enhanced sc-eQTL filtering system with full-scan capabilities and AI-powered content recognition.
    """
    
    def __init__(self, 
                 schema: str = "scagent", 
                 table: str = "merged/sra_geo_ft2",
                 ai_client: Optional[Any] = None):
        self.table_handler = MergedTableHandler(schema, table)
        self.ai_client = ai_client or get_model_client()
        self.filter_cache = {}
        self.processing_stats = defaultdict(int)
        
    def initialize_system(self) -> Dict[str, Any]:
        """
        Initialize the enhanced filtering system with table analysis.
        
        Returns:
            Dict containing initialization results and system recommendations
        """
        logger.info("Initializing enhanced filtering system...")
        
        start_time = time.time()
        
        # Discover table structure
        logger.info("Analyzing table structure...")
        structure = self.table_handler.discover_table_structure()
        
        # Create column mappings
        logger.info("Creating column mappings...")
        mapping = self.table_handler.create_column_mapping()
        
        # Analyze relationships
        logger.info("Analyzing study-run relationships...")
        relationships = self.table_handler.optimize_study_run_relationships()
        
        # Generate summary
        logger.info("Generating table summary...")
        summary = self.table_handler.get_table_summary()
        
        initialization_time = time.time() - start_time
        
        init_results = {
            "status": "success",
            "initialization_time": initialization_time,
            "table_structure": structure,
            "column_mapping": mapping,
            "relationship_analysis": relationships,
            "table_summary": summary,
            "system_ready": True,
            "recommendations": self._generate_system_recommendations(structure, mapping, relationships),
            "initialization_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"System initialized successfully in {initialization_time:.2f} seconds")
        return init_results
    
    def full_scan_filter(self,
                        filter_config: Optional[Dict[str, Any]] = None,
                        batch_size: int = 10000,
                        max_records: Optional[int] = None,
                        enable_ai_assistance: bool = True,
                        enable_parallel: bool = True,
                        output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform full-scan filtering with advanced optimization strategies.
        
        Args:
            filter_config: Enhanced filter configuration
            batch_size: Records per batch for processing
            max_records: Maximum records to process (None for unlimited)
            enable_ai_assistance: Enable AI-powered content recognition
            enable_parallel: Enable parallel processing
            output_file: Output file path for results
            
        Returns:
            Dict containing filtering results and statistics
        """
        logger.info("Starting full-scan filtering...")
        
        # Setup default enhanced filter config
        if filter_config is None:
            filter_config = self._get_enhanced_filter_config()
        
        # Initialize processing statistics
        stats = {
            "total_processed": 0,
            "total_passed": 0,
            "processing_start": datetime.now(),
            "batch_times": [],
            "error_count": 0,
            "ai_assistance_used": 0,
            "field_concatenations": 0,
            "relationship_optimizations": 0
        }
        
        # Results storage
        filtered_results = []
        processing_errors = []
        
        try:
            # Get column mapping for dynamic field handling
            column_mapping = self.table_handler._column_mapping or self.table_handler.create_column_mapping()
            
            # Build filter conditions from config
            conditions = self._build_scan_conditions(filter_config)
            
            # Process in batches
            batch_number = 0
            for batch in self.table_handler.full_scan_query(
                conditions=conditions,
                batch_size=batch_size,
                max_records=max_records,
                enable_parallel=enable_parallel
            ):
                batch_number += 1
                batch_start = time.time()
                
                try:
                    # Process batch with enhanced filtering
                    batch_results = self._process_batch_enhanced(
                        batch, 
                        filter_config, 
                        column_mapping,
                        enable_ai_assistance
                    )
                    
                    # Update statistics
                    stats["total_processed"] += len(batch)
                    stats["total_passed"] += len(batch_results)
                    
                    # Add to results
                    filtered_results.extend(batch_results)
                    
                    batch_time = time.time() - batch_start
                    stats["batch_times"].append(batch_time)
                    
                    # Progress reporting
                    if batch_number % 10 == 0:
                        avg_batch_time = np.mean(stats["batch_times"][-10:])
                        pass_rate = (stats["total_passed"] / stats["total_processed"]) * 100
                        logger.info(f"Batch {batch_number}: {len(batch_results)}/{len(batch)} passed "
                                  f"({pass_rate:.1f}% overall), {avg_batch_time:.2f}s/batch")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_number}: {e}")
                    processing_errors.append({
                        "batch_number": batch_number,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    stats["error_count"] += 1
            
            # Final processing
            stats["processing_end"] = datetime.now()
            stats["total_time"] = (stats["processing_end"] - stats["processing_start"]).total_seconds()
            stats["average_batch_time"] = np.mean(stats["batch_times"]) if stats["batch_times"] else 0
            stats["records_per_second"] = stats["total_processed"] / stats["total_time"] if stats["total_time"] > 0 else 0
            stats["final_pass_rate"] = (stats["total_passed"] / stats["total_processed"]) * 100 if stats["total_processed"] > 0 else 0
            
            # Save results if requested
            if output_file:
                self._save_results(filtered_results, stats, output_file)
            
            # Generate comprehensive report
            report = {
                "scan_summary": {
                    "status": "completed",
                    "total_records_processed": stats["total_processed"],
                    "total_records_passed": stats["total_passed"],
                    "final_pass_rate": stats["final_pass_rate"],
                    "processing_time": stats["total_time"],
                    "records_per_second": stats["records_per_second"]
                },
                "processing_statistics": stats,
                "filter_config": filter_config,
                "sample_results": filtered_results[:10] if filtered_results else [],
                "errors": processing_errors,
                "quality_analysis": self._analyze_result_quality(filtered_results),
                "recommendations": self._generate_post_processing_recommendations(stats, filtered_results),
                "report_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Full-scan completed: {stats['total_passed']}/{stats['total_processed']} records passed "
                       f"({stats['final_pass_rate']:.1f}%) in {stats['total_time']:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Full-scan filtering failed: {e}")
            return {
                "scan_summary": {"status": "failed", "error": str(e)},
                "processing_statistics": stats,
                "report_timestamp": datetime.now().isoformat()
            }
    
    def _get_enhanced_filter_config(self) -> Dict[str, Any]:
        """Get enhanced filter configuration with advanced options."""
        return {
            # Required fields (6)
            "required_fields": {
                "organism": {
                    "required": True,
                    "accepted_values": ["Homo sapiens", "human", "Mus musculus", "mouse"],
                    "fuzzy_matching": True,
                    "ai_inference": True
                },
                "tissue_cell": {
                    "required": True,
                    "exclude_cell_lines": True,
                    "ai_inference": True,
                    "confidence_threshold": 0.7
                },
                "study_accession": {
                    "required": True,
                    "format_validation": True,
                    "uniqueness_check": False
                },
                "run_accession": {
                    "required": True,
                    "format_validation": True,
                    "relationship_validation": True
                },
                "sequencing_method": {
                    "required": True,
                    "accepted_methods": ["RNA-seq", "scRNA-seq", "single-cell RNA-seq", "10x Genomics"],
                    "ai_inference": True
                },
                "sample_size": {
                    "required": True,
                    "min_value": 100,
                    "data_type": "numeric",
                    "ai_estimation": True
                }
            },
            
            # Optional fields (4)
            "optional_fields": {
                "publication": {
                    "boost_score": 2,
                    "ai_inference": True
                },
                "geographic_location": {
                    "boost_score": 1,
                    "standardization": True
                },
                "age_info": {
                    "boost_score": 1,
                    "ai_inference": True
                },
                "disease_annotation": {
                    "boost_score": 2,
                    "exclude_healthy": False,
                    "ai_classification": True
                }
            },
            
            # Processing options
            "processing": {
                "enable_dynamic_concatenation": True,
                "enable_ai_inference": True,
                "enable_relationship_optimization": True,
                "confidence_threshold": 0.6,
                "min_quality_score": 3,
                "max_null_percentage": 70
            },
            
            # AI assistance options
            "ai_assistance": {
                "enabled": True,
                "batch_size": 10,
                "confidence_threshold": 0.7,
                "retry_attempts": 2,
                "timeout": 30
            }
        }
    
    def _build_scan_conditions(self, filter_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build database query conditions from filter configuration."""
        conditions = {}
        
        # Add basic organism filter to reduce scan size
        if "organism" in filter_config.get("required_fields", {}):
            organism_config = filter_config["required_fields"]["organism"]
            if organism_config.get("accepted_values"):
                # Try to map to actual column names
                if self.table_handler._column_mapping:
                    organism_cols = self.table_handler._column_mapping["required_fields"].get("organism", [])
                    if organism_cols:
                        col_name = organism_cols[0]["column_name"]
                        conditions[col_name] = {"like": "sapiens|human|musculus|mouse"}
        
        # Add non-null conditions for critical fields
        if self.table_handler._column_mapping:
            study_cols = self.table_handler._column_mapping["required_fields"].get("study_accession", [])
            run_cols = self.table_handler._column_mapping["required_fields"].get("run_accession", [])
            
            if study_cols:
                conditions[study_cols[0]["column_name"]] = {"null": False}
            if run_cols:
                conditions[run_cols[0]["column_name"]] = {"null": False}
        
        return conditions if conditions else None
    
    def _process_batch_enhanced(self,
                              batch: List[Dict[str, Any]], 
                              filter_config: Dict[str, Any],
                              column_mapping: Dict[str, Any],
                              enable_ai: bool) -> List[Dict[str, Any]]:
        """Process a batch with enhanced filtering strategies."""
        
        results = []
        
        for record in batch:
            try:
                # Step 1: Dynamic field concatenation
                enhanced_record = self._apply_dynamic_concatenation(record, column_mapping)
                
                # Step 2: Apply relationship optimization
                optimized_record = self._apply_relationship_optimization(enhanced_record, column_mapping)
                
                # Step 3: Basic filtering
                basic_result = self._apply_basic_enhanced_filters(optimized_record, filter_config)
                
                if not basic_result["passes_basic"]:
                    continue
                
                # Step 4: AI assistance for uncertain cases
                if enable_ai and basic_result.get("needs_ai_review", False):
                    ai_result = self._apply_ai_assistance(optimized_record, filter_config)
                    if ai_result["ai_decision"] == "reject":
                        continue
                    
                    # Merge AI insights
                    basic_result.update(ai_result)
                
                # Step 5: Calculate final score
                final_score = self._calculate_enhanced_score(basic_result, filter_config)
                
                if final_score >= filter_config["processing"]["min_quality_score"]:
                    result_record = {
                        "original_record": record,
                        "enhanced_record": optimized_record,
                        "filter_result": basic_result,
                        "final_score": final_score,
                        "processing_metadata": {
                            "batch_processed": True,
                            "ai_assisted": basic_result.get("ai_assisted", False),
                            "concatenated_fields": len([k for k in optimized_record.keys() if k.startswith("enhanced_")]),
                            "relationship_optimized": True
                        }
                    }
                    
                    results.append(result_record)
                
            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        return results
    
    def _apply_dynamic_concatenation(self, 
                                   record: Dict[str, Any], 
                                   column_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic field concatenation for NULL values."""
        
        # Build field mapping for concatenation
        field_mapping = {}
        
        # Required fields
        for logical_field, candidates in column_mapping["required_fields"].items():
            if candidates:
                field_mapping[logical_field] = [c["column_name"] for c in candidates[:3]]  # Top 3 candidates
        
        # Optional fields  
        for logical_field, candidates in column_mapping["optional_fields"].items():
            if candidates:
                field_mapping[logical_field] = [c["column_name"] for c in candidates[:2]]  # Top 2 candidates
        
        # Apply concatenation
        enhanced_record = self.table_handler.dynamic_field_concatenation(
            record, 
            field_mapping, 
            use_ai_fallback=True
        )
        
        return enhanced_record
    
    def _apply_relationship_optimization(self, 
                                       record: Dict[str, Any], 
                                       column_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Apply study-run relationship optimization."""
        
        optimized_record = record.copy()
        
        # Get study and run accession fields
        study_cols = column_mapping["required_fields"].get("study_accession", [])
        run_cols = column_mapping["required_fields"].get("run_accession", [])
        
        if study_cols and run_cols:
            study_col = study_cols[0]["column_name"]
            run_col = run_cols[0]["column_name"]
            
            study_value = record.get(study_col)
            run_value = record.get(run_col)
            
            if study_value and run_value:
                # Create relationship metadata
                optimized_record["relationship_metadata"] = {
                    "study_accession": study_value,
                    "run_accession": run_value,
                    "relationship_type": "one_to_many",
                    "is_primary_run": True,  # Could be enhanced with actual analysis
                    "estimated_study_size": 1  # Could be enhanced with cache lookup
                }
        
        return optimized_record
    
    def _apply_basic_enhanced_filters(self, 
                                    record: Dict[str, Any], 
                                    filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhanced basic filtering logic."""
        
        result = {
            "passes_basic": True,
            "filter_scores": {},
            "confidence_scores": {},
            "issues": [],
            "needs_ai_review": False,
            "enhanced_fields_used": []
        }
        
        # Check required fields
        for field_name, field_config in filter_config["required_fields"].items():
            field_result = self._evaluate_enhanced_field(record, field_name, field_config, required=True)
            
            result["filter_scores"][field_name] = field_result["score"]
            result["confidence_scores"][field_name] = field_result["confidence"]
            
            if field_result["enhanced_used"]:
                result["enhanced_fields_used"].append(field_name)
            
            if field_result["score"] == 0:
                result["passes_basic"] = False
                result["issues"].append(f"Required field {field_name} failed: {field_result['reason']}")
            elif field_result["confidence"] < 0.7:
                result["needs_ai_review"] = True
        
        # Check optional fields (for scoring)
        for field_name, field_config in filter_config["optional_fields"].items():
            field_result = self._evaluate_enhanced_field(record, field_name, field_config, required=False)
            
            result["filter_scores"][field_name] = field_result["score"]
            result["confidence_scores"][field_name] = field_result["confidence"]
            
            if field_result["enhanced_used"]:
                result["enhanced_fields_used"].append(field_name)
        
        return result
    
    def _evaluate_enhanced_field(self, 
                               record: Dict[str, Any], 
                               field_name: str, 
                               field_config: Dict[str, Any], 
                               required: bool) -> Dict[str, Any]:
        """Evaluate a single field with enhanced logic."""
        
        result = {
            "score": 0,
            "confidence": 0.0,
            "reason": "",
            "enhanced_used": False,
            "value_found": None
        }
        
        # Look for enhanced field first
        enhanced_key = f"enhanced_{field_name}"
        if enhanced_key in record:
            enhanced_data = record[enhanced_key]
            if enhanced_data and enhanced_data.get("value"):
                result["value_found"] = enhanced_data["value"]
                result["enhanced_used"] = True
                result["confidence"] = 0.8 if enhanced_data["inference_method"] == "concatenation" else 0.6
        
        # If no enhanced field, look for direct matches
        if not result["value_found"]:
            # This would need actual column mapping logic
            # For now, simplified approach
            potential_keys = [k for k in record.keys() if field_name.lower() in k.lower()]
            for key in potential_keys:
                if record[key] is not None:
                    result["value_found"] = record[key]
                    result["confidence"] = 0.9
                    break
        
        # Evaluate the found value
        if result["value_found"]:
            evaluation = self._evaluate_field_value(result["value_found"], field_name, field_config)
            result["score"] = evaluation["score"]
            result["reason"] = evaluation["reason"]
            if evaluation["confidence_modifier"]:
                result["confidence"] *= evaluation["confidence_modifier"]
        else:
            result["reason"] = f"No value found for {field_name}"
            if not required:
                result["score"] = 0  # Neutral for optional fields
        
        return result
    
    def _evaluate_field_value(self, 
                            value: Any, 
                            field_name: str, 
                            field_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific field value against its configuration."""
        
        result = {
            "score": 1,  # Default pass
            "reason": "Value found",
            "confidence_modifier": 1.0
        }
        
        value_str = str(value).lower().strip()
        
        # Field-specific evaluations
        if field_name == "organism":
            accepted = field_config.get("accepted_values", [])
            if accepted:
                if any(acc.lower() in value_str for acc in accepted):
                    result["score"] = 2
                    result["reason"] = f"Organism matches accepted values: {value}"
                else:
                    result["score"] = 0
                    result["reason"] = f"Organism '{value}' not in accepted list"
        
        elif field_name == "tissue_cell":
            if field_config.get("exclude_cell_lines", False):
                cell_line_indicators = ["cell line", "cell-line", "cultured", "immortalized"]
                if any(indicator in value_str for indicator in cell_line_indicators):
                    result["score"] = 0
                    result["reason"] = "Appears to be cell line data"
        
        elif field_name == "sample_size":
            try:
                numeric_value = float(value)
                min_value = field_config.get("min_value", 0)
                if numeric_value >= min_value:
                    result["score"] = 2 if numeric_value >= min_value * 2 else 1
                    result["reason"] = f"Sample size {numeric_value} meets requirements"
                else:
                    result["score"] = 0
                    result["reason"] = f"Sample size {numeric_value} below minimum {min_value}"
            except (ValueError, TypeError):
                result["score"] = 0
                result["reason"] = f"Invalid sample size value: {value}"
                result["confidence_modifier"] = 0.5
        
        elif field_name == "sequencing_method":
            accepted_methods = field_config.get("accepted_methods", [])
            if accepted_methods:
                if any(method.lower() in value_str for method in accepted_methods):
                    result["score"] = 2
                    result["reason"] = f"Sequencing method matches accepted: {value}"
                else:
                    result["score"] = 1
                    result["reason"] = f"Sequencing method present but not optimal: {value}"
                    result["confidence_modifier"] = 0.7
        
        return result
    
    def _apply_ai_assistance(self, 
                           record: Dict[str, Any], 
                           filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AI assistance for uncertain field evaluations."""
        
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(record, filter_config)
            
            # Call AI model
            prompt = self._generate_ai_prompt(context)
            
            response = self.ai_client.chat.completions.create(
                model="Qwen3-235B-A22B",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            ai_analysis = self._parse_ai_response(response.choices[0].message.content)
            
            return {
                "ai_decision": ai_analysis.get("decision", "uncertain"),
                "ai_confidence": ai_analysis.get("confidence", 0.5),
                "ai_reasoning": ai_analysis.get("reasoning", ""),
                "ai_suggested_improvements": ai_analysis.get("improvements", []),
                "ai_assisted": True
            }
            
        except Exception as e:
            logger.warning(f"AI assistance failed: {e}")
            return {
                "ai_decision": "uncertain",
                "ai_confidence": 0.5,
                "ai_reasoning": f"AI assistance failed: {e}",
                "ai_assisted": False
            }
    
    def _prepare_ai_context(self, record: Dict[str, Any], filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context information for AI analysis."""
        
        # Extract relevant fields for AI analysis
        relevant_data = {}
        
        # Get text fields that might contain useful information
        text_fields = ['title', 'summary', 'description', 'abstract', 'study_title']
        for field in text_fields:
            for key, value in record.items():
                if any(tf in key.lower() for tf in text_fields) and isinstance(value, str):
                    relevant_data[key] = value[:500]  # Limit length
        
        # Get enhanced fields
        enhanced_fields = {k: v for k, v in record.items() if k.startswith('enhanced_')}
        
        return {
            "relevant_text_data": relevant_data,
            "enhanced_fields": enhanced_fields,
            "filter_requirements": filter_config["required_fields"],
            "processing_options": filter_config["processing"]
        }
    
    def _generate_ai_prompt(self, context: Dict[str, Any]) -> str:
        """Generate AI prompt for field evaluation."""
        
        prompt = """You are an expert in single-cell RNA-seq data analysis. Please analyze the following dataset record to determine if it's suitable for sc-eQTL analysis.

Required criteria:
1. Organism: Must be human (Homo sapiens) or mouse (Mus musculus)
2. Tissue/Cell type: Must be primary tissue/cells, not cell lines
3. Sequencing method: Must be single-cell RNA-seq or similar
4. Sample size: Should have adequate number of cells (>100)
5. Study/Run accessions: Must have valid identifiers
6. Overall quality: Data should be of sufficient quality for eQTL analysis

Available data:
"""
        
        # Add relevant text data
        if context["relevant_text_data"]:
            prompt += "\nText descriptions:\n"
            for key, value in context["relevant_text_data"].items():
                prompt += f"- {key}: {value}\n"
        
        # Add enhanced fields
        if context["enhanced_fields"]:
            prompt += "\nEnhanced/inferred fields:\n"
            for key, value in context["enhanced_fields"].items():
                if isinstance(value, dict) and "value" in value:
                    prompt += f"- {key}: {value['value']} (confidence: {value.get('inference_method', 'unknown')})\n"
                else:
                    prompt += f"- {key}: {value}\n"
        
        prompt += """
Please provide your assessment in the following JSON format:
{
  "decision": "accept|reject|uncertain",
  "confidence": 0.8,
  "reasoning": "Detailed explanation of your decision",
  "field_assessments": {
    "organism": {"suitable": true, "reason": "..."},
    "tissue_cell": {"suitable": true, "reason": "..."},
    "sequencing_method": {"suitable": true, "reason": "..."},
    "sample_size": {"suitable": true, "reason": "..."}
  },
  "improvements": ["suggestion1", "suggestion2"]
}
"""
        
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response into structured format."""
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "decision": "uncertain",
                    "confidence": 0.5,
                    "reasoning": response_text,
                    "improvements": []
                }
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return {
                "decision": "uncertain", 
                "confidence": 0.3,
                "reasoning": f"Parse error: {e}",
                "improvements": []
            }
    
    def _calculate_enhanced_score(self, 
                                filter_result: Dict[str, Any], 
                                filter_config: Dict[str, Any]) -> float:
        """Calculate enhanced quality score for a record."""
        
        base_score = 0
        bonus_score = 0
        
        # Required fields base score
        required_scores = []
        for field_name in filter_config["required_fields"].keys():
            score = filter_result["filter_scores"].get(field_name, 0)
            confidence = filter_result["confidence_scores"].get(field_name, 1.0)
            weighted_score = score * confidence
            required_scores.append(weighted_score)
        
        base_score = sum(required_scores)
        
        # Optional fields bonus
        for field_name, field_config in filter_config["optional_fields"].items():
            score = filter_result["filter_scores"].get(field_name, 0)
            boost = field_config.get("boost_score", 1)
            bonus_score += score * boost
        
        # AI assistance bonus
        if filter_result.get("ai_assisted") and filter_result.get("ai_confidence", 0) > 0.7:
            bonus_score += 1
        
        # Enhanced fields bonus
        if filter_result.get("enhanced_fields_used"):
            bonus_score += len(filter_result["enhanced_fields_used"]) * 0.5
        
        return base_score + (bonus_score * 0.5)  # Weight bonus at 50%
    
    def _analyze_result_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality distribution of filtering results."""
        
        if not results:
            return {"status": "no_results"}
        
        scores = [r["final_score"] for r in results]
        ai_assisted = sum(1 for r in results if r.get("filter_result", {}).get("ai_assisted", False))
        enhanced_used = sum(1 for r in results if r.get("filter_result", {}).get("enhanced_fields_used", []))
        
        return {
            "total_results": len(results),
            "score_statistics": {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            },
            "ai_assistance_rate": (ai_assisted / len(results)) * 100,
            "enhanced_fields_rate": (enhanced_used / len(results)) * 100,
            "quality_distribution": {
                "high_quality": sum(1 for s in scores if s >= 8),
                "medium_quality": sum(1 for s in scores if 5 <= s < 8),
                "low_quality": sum(1 for s in scores if s < 5)
            }
        }
    
    def _generate_system_recommendations(self, 
                                       structure: Dict, 
                                       mapping: Dict, 
                                       relationships: Dict) -> List[Dict[str, str]]:
        """Generate system-level recommendations based on analysis."""
        
        recommendations = []
        
        # Table size recommendations
        total_records = structure["basic_stats"]["row_count"]
        if total_records > 10000000:  # 10M+ records
            recommendations.append({
                "type": "performance",
                "priority": "critical",
                "title": "Large Dataset Optimization",
                "description": f"With {total_records:,} records, consider using parallel processing and large batch sizes",
                "action": "Enable parallel processing with batch_size >= 50000"
            })
        
        # Column mapping recommendations
        required_unmapped = len([f for f, candidates in mapping["required_fields"].items() if not candidates])
        if required_unmapped > 0:
            recommendations.append({
                "type": "data_quality",
                "priority": "high",
                "title": "Column Mapping Issues",
                "description": f"{required_unmapped} required fields could not be automatically mapped",
                "action": "Review and manually map missing required fields"
            })
        
        # Relationship optimization recommendations
        if "relationship_stats" in relationships:
            multi_run_pct = relationships["relationship_stats"].get("multi_run_percentage", 0)
            if multi_run_pct > 60:
                recommendations.append({
                    "type": "optimization",
                    "priority": "medium", 
                    "title": "Study-Run Relationship Optimization",
                    "description": f"{multi_run_pct:.1f}% of studies have multiple runs",
                    "action": "Enable hierarchical display and study-level filtering"
                })
        
        return recommendations
    
    def _generate_post_processing_recommendations(self, 
                                                stats: Dict, 
                                                results: List[Dict]) -> List[Dict[str, str]]:
        """Generate recommendations after processing completion."""
        
        recommendations = []
        
        # Pass rate analysis
        pass_rate = stats.get("final_pass_rate", 0)
        if pass_rate < 5:
            recommendations.append({
                "type": "filter_tuning",
                "priority": "high", 
                "title": "Low Pass Rate",
                "description": f"Only {pass_rate:.1f}% of records passed filtering",
                "action": "Consider relaxing filter criteria or improving field mapping"
            })
        elif pass_rate > 50:
            recommendations.append({
                "type": "filter_tuning",
                "priority": "medium",
                "title": "High Pass Rate", 
                "description": f"{pass_rate:.1f}% of records passed - filters may be too lenient",
                "action": "Consider tightening quality requirements"
            })
        
        # Performance analysis
        rps = stats.get("records_per_second", 0)
        if rps < 100:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "title": "Processing Speed", 
                "description": f"Processing rate: {rps:.1f} records/second",
                "action": "Consider optimizing batch size or enabling parallel processing"
            })
        
        return recommendations
    
    def _save_results(self, results: List[Dict[str, Any]], stats: Dict[str, Any], output_file: str):
        """Save filtering results to file."""
        
        output_data = {
            "metadata": {
                "total_results": len(results),
                "processing_stats": stats,
                "generated_at": datetime.now().isoformat(),
                "scagent_version": "2.0_enhanced"
            },
            "results": results
        }
        
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
            elif output_file.endswith('.csv'):
                # Flatten results for CSV
                flattened_results = []
                for result in results:
                    flat_record = result["original_record"].copy()
                    flat_record.update({
                        "final_score": result["final_score"],
                        "ai_assisted": result.get("filter_result", {}).get("ai_assisted", False),
                        "enhanced_fields_count": len(result.get("filter_result", {}).get("enhanced_fields_used", []))
                    })
                    flattened_results.append(flat_record)
                
                import pandas as pd
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_file, index=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

import re
from typing import Dict, Any, List, Optional

# --- Enhanced Human Species Identification ---

def is_human_sample_enhanced(record: Dict[str, Any], column_mapping: Dict[str, str]) -> bool:
    """
    Enhanced check to determine if a sample is of human origin.
    Handles mixed organism entries like "Homo sapiens; Mus musculus" by checking if Homo sapiens is present.

    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.

    Returns:
        bool: True if the sample is determined to be human, False otherwise.
    """
    # Human synonyms to check for
    human_keywords = ['homo sapiens', 'human', 'h. sapiens', 'hsapiens']
    
    # Prefer 'organism_ch1' as it is more reliable
    organism_ch1_col = column_mapping.get("organism_ch1")
    if organism_ch1_col and organism_ch1_col in record:
        organism_ch1_val = str(record[organism_ch1_col] or '').lower().strip()
        if organism_ch1_val and any(keyword in organism_ch1_val for keyword in human_keywords):
            return True

    # Fallback to 'experiment_title' - parse format like "GSM5163384: FSE_RAS_day4_2; Homo sapiens; RNA-Seq"
    experiment_title_col = column_mapping.get("experiment_title")
    if experiment_title_col and experiment_title_col in record:
        experiment_title_val = str(record[experiment_title_col] or '').lower().strip()
        if experiment_title_val:
            # Split by semicolon to handle structured experiment titles
            title_parts = [part.strip() for part in experiment_title_val.split(';')]
            for part in title_parts:
                if any(keyword in part for keyword in human_keywords):
                    return True
    
    # Skip 'organism' column as it has too many NULLs (as per user requirement)
    # organism_col = column_mapping.get("organism")
    # if organism_col and organism_col in record:
    #     organism_val = str(record[organism_col] or '').lower().strip()
    #     if organism_val and any(keyword in organism_val for keyword in human_keywords):
    #         return True

    return False

# --- Cell Line Exclusion ---

def is_cell_line_sample(record: Dict[str, Any], column_mapping: Dict[str, str]) -> bool:
    """
    Checks if a sample is derived from a cell line based on 'characteristics_ch1'.

    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.

    Returns:
        bool: True if it is a cell line, False otherwise.
    """
    characteristics_col = column_mapping.get("characteristics_ch1")
    if characteristics_col and characteristics_col in record:
        characteristics_val = str(record[characteristics_col]).lower()
        # Common patterns indicating a cell line
        if 'cell line:' in characteristics_val or 'cell line name:' in characteristics_val:
            return True
    return False

# --- Single-Cell Experiment Identification ---

def is_single_cell_experiment(record: Dict[str, Any], column_mapping: Dict[str, str]) -> bool:
    """
    Identifies if an experiment is single-cell based on various text fields.
    Specifically looks for scRNA-seq and scATAC-seq experiments.

    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.

    Returns:
        bool: True if it is a single-cell experiment, False otherwise.
    """
    # Keywords for single-cell technologies (prioritized)
    sc_keywords = [
        'scrna-seq', 'scrna seq', 'sc-rna-seq', 'sc rna seq', 'sc-rna',
        'scatac-seq', 'scatac seq', 'sc-atac-seq', 'sc atac seq', 'sc-atac',
        'single-cell rna', 'single cell rna', 'single-cell atac', 'single cell atac',
        'single-cell', 'single cell', 'sc seq', 'sc-seq',
        '10x genomics', '10x chromium', 'chromium', 'drop-seq', 'dropseq',
        'smart-seq', 'smartseq', 'cel-seq', 'celseq', 'in-drop', 'indrop',
        'mars-seq', 'marsseq', 'plate-seq', 'plateseq'
    ]
    
    # Fields to search for keywords (prioritized order)
    fields_to_check = [
        "experiment_title", "study_title", "study_abstract", 
        "library_strategy", "library_source", "library_selection",
        "instrument_model", "platform", "title", "summary"
    ]

    for field_key in fields_to_check:
        column_name = column_mapping.get(field_key)
        if column_name and column_name in record:
            content = str(record[column_name] or '').lower().strip()
            if content:
                # Special handling for experiment_title format like "GSM4253928: H2MGEMD14; Homo sapiens; RNA-Seq"
                if field_key == "experiment_title":
                    # Check if it contains RNA-Seq but also has single-cell indicators
                    if 'rna-seq' in content or 'rna seq' in content:
                        # Look for single-cell indicators in the same title
                        for keyword in sc_keywords:
                            if keyword in content:
                                return True
                        # Also check if it's part of a larger study that might be single-cell
                        # by looking at other fields
                        continue
                
                # Direct keyword matching
                if any(keyword in content for keyword in sc_keywords):
                    return True
    
    # Additional check: if library_strategy is RNA-Seq, check other fields for single-cell indicators
    library_strategy_col = column_mapping.get("library_strategy")
    if library_strategy_col and library_strategy_col in record:
        lib_strategy = str(record[library_strategy_col] or '').lower().strip()
        if 'rna-seq' in lib_strategy or 'rna seq' in lib_strategy:
            # Check study title and abstract for single-cell indicators
            for field_key in ["study_title", "study_abstract", "title", "summary"]:
                column_name = column_mapping.get(field_key)
                if column_name and column_name in record:
                    content = str(record[column_name] or '').lower().strip()
                    if content and any(keyword in content for keyword in sc_keywords):
                        return True
                
    return False

# --- Comprehensive Pre-Filter ---

def pre_filter_record(record: Dict[str, Any], column_mapping: Dict[str, str]) -> Optional[str]:
    """
    Applies a series of pre-filters to a record to quickly exclude non-relevant data.

    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.

    Returns:
        Optional[str]: A reason for rejection if filtered out, otherwise None.
    """
    # 1. Must be a human sample
    if not is_human_sample_enhanced(record, column_mapping):
        return "Not a human sample"

    # 2. Must not be a cell line
    if is_cell_line_sample(record, column_mapping):
        return "Sample is a cell line"

    # 3. Must be a single-cell experiment
    if not is_single_cell_experiment(record, column_mapping):
        return "Not a single-cell experiment"

    return None

# --- Additional Information Extraction ---

def extract_instrument_model(record: Dict[str, Any], column_mapping: Dict[str, str]) -> str:
    """
    Extract instrument model information from the record.

    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.

    Returns:
        str: Instrument model if found, empty string otherwise.
    """
    # Check instrument_model column first
    instrument_col = column_mapping.get("instrument_model")
    if instrument_col and instrument_col in record:
        instrument_val = str(record[instrument_col] or '').strip()
        if instrument_val and instrument_val.lower() not in ['', 'null', 'none', 'na']:
            return instrument_val
    
    # Fallback to platform column
    platform_col = column_mapping.get("platform")
    if platform_col and platform_col in record:
        platform_val = str(record[platform_col] or '').strip()
        if platform_val and platform_val.lower() not in ['', 'null', 'none', 'na']:
            return platform_val
    
    # Check experiment_title for instrument information
    experiment_title_col = column_mapping.get("experiment_title")
    if experiment_title_col and experiment_title_col in record:
        title_val = str(record[experiment_title_col] or '').lower()
        
        # Common single-cell instruments
        instruments = [
            'illumina hiseq', 'illumina nextseq', 'illumina novaseq', 'illumina miseq',
            '10x genomics', '10x chromium', 'chromium', 'pacbio', 'oxford nanopore'
        ]
        
        for instrument in instruments:
            if instrument in title_val:
                return instrument.title()
    
    return ""

def extract_sequencing_strategy(record: Dict[str, Any], column_mapping: Dict[str, str]) -> str:
    """
    Extract sequencing strategy information (scRNA-seq, scATAC-seq, etc.).

    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.

    Returns:
        str: Sequencing strategy if identified, empty string otherwise.
    """
    # Strategy mapping
    strategy_patterns = {
        'scRNA-seq': ['scrna-seq', 'scrna seq', 'sc-rna-seq', 'sc rna seq', 'single-cell rna'],
        'scATAC-seq': ['scatac-seq', 'scatac seq', 'sc-atac-seq', 'sc atac seq', 'single-cell atac'],
        'Single-cell': ['single-cell', 'single cell', 'sc seq', 'sc-seq']
    }
    
    # Fields to check
    fields_to_check = ["experiment_title", "study_title", "study_abstract", "library_strategy", "title", "summary"]
    
    for field_key in fields_to_check:
        column_name = column_mapping.get(field_key)
        if column_name and column_name in record:
            content = str(record[column_name] or '').lower().strip()
            if content:
                for strategy, patterns in strategy_patterns.items():
                    if any(pattern in content for pattern in patterns):
                        return strategy
    
    return ""

def extract_sc_eqtl_criteria_with_ai(record: Dict[str, Any], column_mapping: Dict[str, str], ai_client=None) -> Dict[str, Any]:
    """
    Use AI to extract 10 key sc-eQTL criteria from study_title and study_abstract.
    
    Args:
        record (Dict[str, Any]): The dataset record.
        column_mapping (Dict[str, str]): Mapping of logical to physical column names.
        ai_client: AI client for intelligent extraction (optional).
    
    Returns:
        Dict containing extracted criteria information.
    """
    # The 10 key sc-eQTL criteria to extract
    criteria_template = {
        "organism": "",
        "tissue_type": "",
        "cell_type": "",
        "sample_size": "",
        "platform": "",
        "project_id": "",
        "publication": "",
        "geographic_location": "",
        "age_range": "",
        "disease_status": ""
    }
    
    # Collect text for analysis
    study_title_col = column_mapping.get("study_title")
    study_abstract_col = column_mapping.get("study_abstract")
    
    study_title = str(record.get(study_title_col, '') or '') if study_title_col else ''
    study_abstract = str(record.get(study_abstract_col, '') or '') if study_abstract_col else ''
    
    # Combine text for analysis
    combined_text = f"Title: {study_title}\nAbstract: {study_abstract}".strip()
    
    if not combined_text or combined_text == "Title: \nAbstract: ":
        return criteria_template
    
    # If AI client is available, use it for intelligent extraction
    if ai_client:
        try:
            prompt = f"""
            Analyze the following study information and extract key sc-eQTL relevant criteria:

            {combined_text}

            Please extract the following information if available:
            1. Organism (species)
            2. Tissue type
            3. Cell type
            4. Sample size (number of samples/cells)
            5. Platform (sequencing platform)
            6. Project ID (GEO, SRA accessions)
            7. Publication (PMID, DOI)
            8. Geographic location
            9. Age range
            10. Disease status

            Return only the extracted values in a structured format, use "Not specified" if information is not available.
            """
            
            response = ai_client.generate(prompt, temperature=0.3)
            # Parse AI response (simplified - would need more robust parsing)
            # For now, return template with some basic extraction
            
        except Exception as e:
            # Fall back to rule-based extraction if AI fails
            pass
    
    # Rule-based extraction as fallback
    combined_lower = combined_text.lower()
    
    # Extract organism
    if 'homo sapiens' in combined_lower or 'human' in combined_lower:
        criteria_template["organism"] = "Homo sapiens"
    elif 'mus musculus' in combined_lower or 'mouse' in combined_lower:
        criteria_template["organism"] = "Mus musculus"
    
    # Extract tissue type (basic patterns)
    tissues = ['brain', 'liver', 'heart', 'lung', 'kidney', 'muscle', 'blood', 'skin', 'bone', 'pancreas']
    for tissue in tissues:
        if tissue in combined_lower:
            criteria_template["tissue_type"] = tissue.title()
            break
    
    # Extract cell type (basic patterns)
    cell_types = ['neuron', 'hepatocyte', 'fibroblast', 'endothelial', 'epithelial', 'immune', 'stem cell']
    for cell_type in cell_types:
        if cell_type in combined_lower:
            criteria_template["cell_type"] = cell_type.title()
            break
    
    # Extract sample size (look for numbers followed by relevant keywords)
    import re
    sample_patterns = [
        r'(\d+)\s*(?:samples?|cells?|patients?|subjects?)',
        r'n\s*=\s*(\d+)',
        r'(\d+)\s*(?:single.?cells?)'
    ]
    
    for pattern in sample_patterns:
        match = re.search(pattern, combined_lower)
        if match:
            criteria_template["sample_size"] = match.group(1)
            break
    
    # Extract platform
    platforms = ['10x genomics', 'illumina', 'smart-seq', 'drop-seq', 'chromium']
    for platform in platforms:
        if platform in combined_lower:
            criteria_template["platform"] = platform
            break
    
    return criteria_template

def test_enhanced_filtering():
    """Test function for enhanced filtering system."""
    
    print("Testing Enhanced sc-eQTL Filtering System...")
    
    try:
        # Initialize system
        filter_system = EnhancedScEQTLFilter()
        
        print("1. Initializing system...")
        init_results = filter_system.initialize_system()
        if init_results["status"] == "success":
            print(f"✓ System initialized in {init_results['initialization_time']:.2f}s")
            print(f"✓ Found {init_results['table_structure']['total_columns']} columns")
        else:
            print("✗ System initialization failed")
            return False
        
        # Test small scan
        print("2. Testing small-scale filtering...")
        scan_results = filter_system.full_scan_filter(
            max_records=1000,
            batch_size=100,
            enable_ai_assistance=False  # Disable for faster testing
        )
        
        if scan_results["scan_summary"]["status"] == "completed":
            total_processed = scan_results["scan_summary"]["total_records_processed"]  
            total_passed = scan_results["scan_summary"]["total_records_passed"]
            pass_rate = scan_results["scan_summary"]["final_pass_rate"]
            print(f"✓ Processed {total_processed} records, {total_passed} passed ({pass_rate:.1f}%)")
        else:
            print("✗ Small-scale filtering failed")
            return False
        
        print("✓ Enhanced filtering system test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def quick_table_inspection():
    """
    简洁的表检索和查询功能
    """
    from .db.connect import get_connection
    import psycopg2.extras
    
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. 查看所有schema
            print("=== Available Schemas ===")
            cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')")
            schemas = cur.fetchall()
            for schema in schemas:
                print(f"- {schema['schema_name']}")
            
            # 2. 查看merged schema下的表
            print("\n=== Tables in 'merged' schema ===")
            cur.execute("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_schema = 'merged' AND table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'merged'
                ORDER BY table_name
            """)
            tables = cur.fetchall()
            
            target_table = None
            for table in tables:
                print(f"- {table['table_name']} ({table['column_count']} columns)")
                if 'sra_geo' in table['table_name'].lower():
                    target_table = table['table_name']
            
            # 3. 如果找到目标表，显示详细信息
            if target_table:
                print(f"\n=== Target Table: merged.{target_table} ===")
                
                # 获取行数
                cur.execute(f"SELECT COUNT(*) as total_rows FROM merged.{target_table}")
                count = cur.fetchone()
                print(f"Total rows: {count['total_rows']:,}")
                
                # 获取前几列名
                cur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'merged' AND table_name = '{target_table}'
                    ORDER BY ordinal_position
                    LIMIT 10
                """)
                columns = cur.fetchall()
                print("First 10 columns:")
                for col in columns:
                    print(f"  - {col['column_name']} ({col['data_type']})")
                
                # 获取样本数据
                print(f"\n=== Sample Data ===")
                cur.execute(f"SELECT * FROM merged.{target_table} LIMIT 3")
                samples = cur.fetchall()
                
                if samples:
                    # 只显示前5个字段的数据
                    first_cols = list(samples[0].keys())[:5]
                    print(f"Showing first 5 columns: {first_cols}")
                    for i, sample in enumerate(samples, 1):
                        print(f"Row {i}:")
                        for col in first_cols:
                            value = str(sample[col])[:50] + "..." if len(str(sample[col])) > 50 else sample[col]
                            print(f"  {col}: {value}")
                        print()
                
                return target_table
            else:
                print("No sra_geo table found")
                return None
                
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_enhanced_filtering() 