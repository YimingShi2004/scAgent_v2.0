#!/usr/bin/env python3
"""
Fixed filter report generation with improved statistics calculation
"""

from typing import Dict, List, Any
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def generate_filter_report_fixed(
    original_records: List[Dict[str, Any]],
    filtered_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate comprehensive filtering report with improved statistics."""
    
    report = {
        "filtering_summary": {
            "original_count": len(original_records),
            "filtered_count": len(filtered_records),
            "retention_rate": (len(filtered_records) / len(original_records)) * 100 if original_records else 0
        },
        "filter_statistics": {},
        "quality_distribution": {},
        "common_rejection_reasons": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Analyze filter statistics
    filter_categories = [
        "species", "cell_line", "database_id", "publication", "sample_size",
        "country", "age", "tumor", "sequencing_method", "tissue"
    ]
    
    for category in filter_categories:
        scores = []
        confidences = []
        passes = []
        
        for record in original_records:
            filter_result = record.get("sc_eqtl_filter_result", {})
            
            # Try multiple ways to get the score
            score = 0
            confidence = 0.0
            
            # Method 1: filter_details (new intelligent filtering)
            if "filter_details" in filter_result and category in filter_result["filter_details"]:
                detail = filter_result["filter_details"][category]
                score = detail.get("score", 0)
                confidence = detail.get("confidence", 0.0)
            
            # Method 2: filter_scores (legacy)
            elif "filter_scores" in filter_result and category in filter_result["filter_scores"]:
                score = filter_result["filter_scores"][category]
                confidence = 1.0  # Assume high confidence for legacy scores
            
            # Method 3: Direct assessment for specific fields
            else:
                # Try to assess directly from record data
                score, confidence = assess_filter_from_record(record, category)
            
            scores.append(score)
            confidences.append(confidence)
            passes.append(score > 0)
        
        if scores:
            report["filter_statistics"][category] = {
                "average_score": sum(scores) / len(scores),
                "average_confidence": sum(confidences) / len(confidences),
                "pass_count": sum(passes),
                "fail_count": len(scores) - sum(passes),
                "pass_rate": (sum(passes) / len(scores)) * 100,
                "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
                "uncertain_count": sum(1 for c in confidences if c < 0.7)
            }
    
    # Analyze quality distribution
    quality_scores = []
    decisions = []
    phases = []
    
    for record in filtered_records:
        filter_result = record.get("sc_eqtl_filter_result", {})
        score = filter_result.get("overall_score", 0)
        decision = filter_result.get("decision", "unknown")
        phase = filter_result.get("phase", "unknown")
        
        quality_scores.append(score)
        decisions.append(decision)
        phases.append(phase)
    
    if quality_scores:
        report["quality_distribution"] = {
            "average_score": sum(quality_scores) / len(quality_scores),
            "min_score": min(quality_scores),
            "max_score": max(quality_scores),
            "high_quality_count": sum(1 for s in quality_scores if s >= 8),
            "medium_quality_count": sum(1 for s in quality_scores if 4 <= s < 8),
            "low_quality_count": sum(1 for s in quality_scores if s < 4)
        }
        
        # Decision distribution
        decision_counts = Counter(decisions)
        report["decision_distribution"] = dict(decision_counts)
        
        # Phase distribution
        phase_counts = Counter(phases)
        report["phase_distribution"] = dict(phase_counts)
    
    # Common rejection reasons
    rejection_reasons = []
    for record in original_records:
        filter_result = record.get("sc_eqtl_filter_result", {})
        
        # Check if record was rejected
        if record not in filtered_records:
            rejection_reason = filter_result.get("rejection_reason")
            if rejection_reason:
                rejection_reasons.append(rejection_reason)
            
            # Also collect reasons from filter details
            if "filter_details" in filter_result:
                for filter_name, detail in filter_result["filter_details"].items():
                    if detail.get("score", 0) == 0 and detail.get("confidence", 0) >= 0.7:
                        reason = f"Failed {filter_name}: {', '.join(detail.get('reasons', []))}"
                        rejection_reasons.append(reason)
    
    reason_counts = Counter(rejection_reasons)
    report["common_rejection_reasons"] = [
        {"reason": reason, "count": count}
        for reason, count in reason_counts.most_common(10)
    ]
    
    return report

def assess_filter_from_record(record: Dict[str, Any], category: str) -> tuple[int, float]:
    """Assess filter score directly from record data when filter_result is missing."""
    
    if category == "species":
        # Check for human species
        organism_fields = [
            record.get("organism", ""),
            record.get("scientific_name", ""),
            record.get("sra_organism", ""),
            record.get("geo_organism", "")
        ]
        
        for field in organism_fields:
            if field and "homo sapiens" in field.lower():
                return 3, 0.9  # High score, high confidence
            elif field and "human" in field.lower():
                return 2, 0.8  # Medium score, good confidence
        
        return 0, 0.8  # No human found, high confidence
    
    elif category == "database_id":
        # Check for database IDs
        id_fields = [
            record.get("sra_run_accession", ""),
            record.get("run_accession", ""),
            record.get("geo_accession", ""),
            record.get("gse", ""),
            record.get("sra_study_accession", ""),
            record.get("study_accession", "")
        ]
        
        valid_ids = [f for f in id_fields if f and len(f) > 3]
        if valid_ids:
            return 3, 0.9  # Has valid ID, high confidence
        return 0, 0.9  # No valid ID, high confidence
    
    elif category == "cell_line":
        # Check for cell line indicators
        text_fields = [
            record.get("study_title", ""),
            record.get("geo_title", ""),
            record.get("summary", ""),
            record.get("geo_summary", "")
        ]
        
        cell_line_indicators = ["cell line", "hela", "293t", "jurkat", "k562"]
        
        for text in text_fields:
            if text:
                text_lower = text.lower()
                for indicator in cell_line_indicators:
                    if indicator in text_lower:
                        return 0, 0.8  # Cell line detected, reject
        
        return 3, 0.6  # Probably not cell line, medium confidence
    
    elif category == "sequencing_method":
        # Check for single-cell sequencing methods
        method_fields = [
            record.get("library_strategy", ""),
            record.get("sra_library_strategy", ""),
            record.get("platform", ""),
            record.get("sra_platform", "")
        ]
        
        sc_methods = ["rna-seq", "single cell", "10x", "drop-seq", "smart-seq"]
        
        for field in method_fields:
            if field:
                field_lower = field.lower()
                for method in sc_methods:
                    if method in field_lower:
                        return 2, 0.8  # Good sequencing method
        
        return 1, 0.5  # Unknown method, low confidence
    
    else:
        # For other categories, return neutral score with low confidence
        return 1, 0.3

if __name__ == "__main__":
    # Test the fixed report generation
    sample_record = {
        "organism": "Homo sapiens",
        "run_accession": "SRR123456",
        "study_title": "Single cell RNA-seq of human tissue",
        "sc_eqtl_filter_result": {
            "overall_score": 7,
            "decision": "accept_conservative",
            "phase": "confidence_assessment",
            "filter_details": {
                "species": {"score": 3, "confidence": 0.9},
                "database_id": {"score": 3, "confidence": 0.9}
            }
        }
    }
    
    report = generate_filter_report_fixed([sample_record], [sample_record])
    print("Generated report:", report["filter_statistics"]) 