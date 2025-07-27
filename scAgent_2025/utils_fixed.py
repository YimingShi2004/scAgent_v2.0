#!/usr/bin/env python3
"""
Fixed utility functions for scAgent with proper field mapping
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def create_integrated_dataset_table_fixed(
    geo_records: List[Dict[str, Any]],
    sra_records: List[Dict[str, Any]],
    mapping: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create integrated dataset table with GEO-SRA relationships.
    Fixed version with proper field mapping.
    
    Args:
        geo_records: List of GEO records
        sra_records: List of SRA records  
        mapping: GEO-SRA mapping information
        
    Returns:
        Integrated dataset table
    """
    integrated_table = []
    
    logger.info(f"Creating integrated dataset from {len(geo_records)} GEO and {len(sra_records)} SRA records")
    
    # Create lookup dictionaries with correct field names
    geo_lookup = {}
    for record in geo_records:
        gse_id = record.get('gse', record.get('geo_accession', ''))
        if gse_id:
            geo_lookup[gse_id] = record
    
    sra_lookup = {}
    for record in sra_records:
        run_id = record.get('run_accession', '')
        if run_id:
            sra_lookup[run_id] = record
    
    logger.info(f"Created lookups: {len(geo_lookup)} GEO, {len(sra_lookup)} SRA")
    
    # Process mapped GEO-SRA pairs
    mapped_count = 0
    for geo_acc, sra_list in mapping.get("geo_to_sra", {}).items():
        geo_record = geo_lookup.get(geo_acc, {})
        
        for sra_acc in sra_list:
            sra_record = sra_lookup.get(sra_acc, {})
            
            if sra_record:  # Only create record if SRA data exists
                integrated_record = create_integrated_record(geo_record, sra_record, "geo_to_sra")
                integrated_table.append(integrated_record)
                mapped_count += 1
    
    # Process orphaned SRA records (SRA-only)
    sra_only_count = 0
    mapped_sra_ids = set()
    for sra_list in mapping.get("geo_to_sra", {}).values():
        mapped_sra_ids.update(sra_list)
    
    for sra_acc, sra_record in sra_lookup.items():
        if sra_acc not in mapped_sra_ids:
            # Create SRA-only record with proper field mapping
            integrated_record = create_sra_only_record(sra_record)
            integrated_table.append(integrated_record)
            sra_only_count += 1
    
    # Process orphaned GEO records (GEO-only)
    geo_only_count = 0
    mapped_geo_ids = set(mapping.get("geo_to_sra", {}).keys())
    
    for geo_acc, geo_record in geo_lookup.items():
        if geo_acc not in mapped_geo_ids:
            # Create GEO-only record
            integrated_record = create_geo_only_record(geo_record)
            integrated_table.append(integrated_record)
            geo_only_count += 1
    
    logger.info(f"Integrated dataset created: {len(integrated_table)} total records "
               f"({mapped_count} mapped, {sra_only_count} SRA-only, {geo_only_count} GEO-only)")
    
    return integrated_table

def create_sra_only_record(sra_record: Dict[str, Any]) -> Dict[str, Any]:
    """Create an integrated record from SRA data only with proper field mapping."""
    
    return {
        # Primary identifiers
        "sra_run_accession": sra_record.get('run_accession', ''),
        "run_accession": sra_record.get('run_accession', ''),  # For compatibility
        "sra_study_accession": sra_record.get('study_accession', ''),
        "study_accession": sra_record.get('study_accession', ''),  # For compatibility
        
        # Species information (multiple field names for compatibility)
        "scientific_name": sra_record.get('scientific_name', ''),
        "organism": sra_record.get('scientific_name', ''),  # Map scientific_name to organism
        "sra_organism": sra_record.get('scientific_name', ''),
        "common_name": sra_record.get('common_name', ''),
        "taxon_id": sra_record.get('taxon_id', ''),
        
        # Title/summary information
        "sra_study_title": sra_record.get('study_title', ''),
        "study_title": sra_record.get('study_title', ''),  # For compatibility
        "geo_title": sra_record.get('study_title', ''),  # For compatibility
        "study_abstract": sra_record.get('study_abstract', ''),
        "geo_summary": sra_record.get('study_abstract', ''),  # For compatibility
        "summary": sra_record.get('study_abstract', ''),  # For compatibility
        
        # Technical details
        "sra_platform": sra_record.get('platform', ''),
        "platform": sra_record.get('platform', ''),  # For compatibility
        "sra_instrument": sra_record.get('instrument_model', ''),
        "instrument_model": sra_record.get('instrument_model', ''),
        "sra_library_strategy": sra_record.get('library_strategy', ''),
        "library_strategy": sra_record.get('library_strategy', ''),
        "sra_library_layout": sra_record.get('library_layout', ''),
        "library_layout": sra_record.get('library_layout', ''),
        "sra_spots": safe_int_convert(sra_record.get('spots', 0)),
        "spots": safe_int_convert(sra_record.get('spots', 0)),
        "sra_bases": safe_int_convert(sra_record.get('bases', 0)),
        "bases": safe_int_convert(sra_record.get('bases', 0)),
        
        # Additional SRA fields
        "design_description": sra_record.get('design_description', ''),
        "sample_name": sra_record.get('sample_name', ''),
        "description": sra_record.get('description', ''),
        "library_source": sra_record.get('library_source', ''),
        "library_selection": sra_record.get('library_selection', ''),
        
        # Empty GEO fields
        "geo_accession": "",
        "gse": "",
        "geo_organism": "",
        "geo_platform": "",
        "geo_sample_count": 0,
        "geo_submission_date": "",
        "geo_status": "",
        
        # Metadata
        "relationship_type": "sra_only",
        "mapping_confidence": 1.0,
        "data_source": "SRA",
        "data_completeness": calculate_sra_completeness(sra_record),
        "recommended_for_eqtl": "Recommended" if is_good_for_eqtl(sra_record) else "Not Recommended",
        
        # Download information
        "fastq_download_url": f"https://www.ncbi.nlm.nih.gov/sra/{sra_record.get('run_accession', '')}",
        "fastq_dump_command": f"fastq-dump --split-files --gzip {sra_record.get('run_accession', '')}",
        "prefetch_command": f"prefetch {sra_record.get('run_accession', '')}",
        "fasterq_dump_command": f"fasterq-dump --split-files {sra_record.get('run_accession', '')}",
        "aspera_download_command": generate_aspera_command(sra_record.get('run_accession', '')),
        "estimated_file_size_gb": estimate_file_size_gb(sra_record)
    }

def create_geo_only_record(geo_record: Dict[str, Any]) -> Dict[str, Any]:
    """Create an integrated record from GEO data only."""
    
    return {
        # GEO identifiers
        "geo_accession": geo_record.get('gse', ''),
        "gse": geo_record.get('gse', ''),  # For compatibility
        
        # Species information
        "organism": geo_record.get('organism', ''),
        "geo_organism": geo_record.get('organism', ''),
        
        # Title/summary
        "geo_title": geo_record.get('gse_title', ''),
        "gse_title": geo_record.get('gse_title', ''),  # For compatibility
        "study_title": geo_record.get('gse_title', ''),  # For compatibility
        "geo_summary": geo_record.get('summary', ''),
        "summary": geo_record.get('summary', ''),  # For compatibility
        
        # Platform
        "geo_platform": geo_record.get('platform', ''),
        "platform": geo_record.get('platform', ''),  # For compatibility
        
        # Dates
        "geo_submission_date": geo_record.get('submission_date', ''),
        "geo_status": geo_record.get('status', ''),
        
        # Empty SRA fields
        "sra_run_accession": "",
        "run_accession": "",
        "sra_study_accession": "",
        "study_accession": "",
        "scientific_name": "",
        "sra_organism": "",
        "common_name": "",
        "taxon_id": "",
        "sra_study_title": "",
        "study_abstract": "",
        "sra_platform": "",
        "sra_instrument": "",
        "sra_library_strategy": "",
        "sra_library_layout": "",
        "sra_spots": 0,
        "sra_bases": 0,
        
        # Metadata
        "relationship_type": "geo_only",
        "mapping_confidence": 0.5,
        "data_source": "GEO",
        "data_completeness": 0.3,  # Lower completeness for GEO-only
        "recommended_for_eqtl": "Not Recommended",  # GEO-only usually not sufficient
        
        # Empty download information
        "fastq_download_url": "",
        "fastq_dump_command": "",
        "prefetch_command": "",
        "fasterq_dump_command": "",
        "aspera_download_command": "",
        "estimated_file_size_gb": 0
    }

def create_integrated_record(
    geo_record: Dict[str, Any], 
    sra_record: Dict[str, Any], 
    relationship_type: str
) -> Dict[str, Any]:
    """Create an integrated record from both GEO and SRA data."""
    
    # Start with SRA record as base (it has the actual data)
    integrated = create_sra_only_record(sra_record)
    
    # Overlay GEO information
    integrated.update({
        "geo_accession": geo_record.get('gse', ''),
        "gse": geo_record.get('gse', ''),
        "geo_title": geo_record.get('gse_title', ''),
        "geo_summary": geo_record.get('summary', ''),
        "geo_organism": geo_record.get('organism', ''),
        "geo_platform": geo_record.get('platform', ''),
        "geo_submission_date": geo_record.get('submission_date', ''),
        "geo_status": geo_record.get('status', ''),
        
        # Update metadata
        "relationship_type": relationship_type,
        "mapping_confidence": calculate_mapping_confidence(geo_record, sra_record),
        "data_completeness": calculate_combined_completeness(geo_record, sra_record),
        "recommended_for_eqtl": "Highly Recommended" if is_good_combined_record(geo_record, sra_record) else "Recommended"
    })
    
    return integrated

def safe_int_convert(value: Any) -> int:
    """Safely convert value to integer."""
    if value is None:
        return 0
    try:
        if isinstance(value, str):
            # Remove commas and convert
            value = value.replace(',', '')
        return int(float(value))
    except (ValueError, TypeError):
        return 0

def calculate_sra_completeness(sra_record: Dict[str, Any]) -> float:
    """Calculate completeness score for SRA record."""
    required_fields = ['run_accession', 'study_title', 'platform', 'library_strategy']
    present_fields = sum(1 for field in required_fields if sra_record.get(field))
    return present_fields / len(required_fields)

def calculate_mapping_confidence(geo_record: Dict[str, Any], sra_record: Dict[str, Any]) -> float:
    """Calculate confidence score for GEO-SRA mapping."""
    # Simple similarity check based on titles
    geo_title = (geo_record.get('gse_title', '') or '').lower()
    sra_title = (sra_record.get('study_title', '') or '').lower()
    
    if geo_title and sra_title:
        # Simple word overlap
        geo_words = set(geo_title.split())
        sra_words = set(sra_title.split())
        if geo_words and sra_words:
            overlap = len(geo_words.intersection(sra_words))
            total = len(geo_words.union(sra_words))
            return overlap / total if total > 0 else 0.5
    
    return 0.5  # Default moderate confidence

def calculate_combined_completeness(geo_record: Dict[str, Any], sra_record: Dict[str, Any]) -> float:
    """Calculate completeness for combined GEO-SRA record."""
    sra_completeness = calculate_sra_completeness(sra_record)
    
    # Bonus for having GEO metadata
    geo_bonus = 0.1 if geo_record.get('gse') else 0
    
    return min(1.0, sra_completeness + geo_bonus)

def is_good_for_eqtl(sra_record: Dict[str, Any]) -> bool:
    """Check if SRA record is good for eQTL analysis."""
    # Simple heuristics
    has_run_id = bool(sra_record.get('run_accession'))
    has_title = bool(sra_record.get('study_title'))
    has_spots = safe_int_convert(sra_record.get('spots', 0)) > 1000
    
    return has_run_id and has_title and has_spots

def is_good_combined_record(geo_record: Dict[str, Any], sra_record: Dict[str, Any]) -> bool:
    """Check if combined GEO-SRA record is good for eQTL analysis."""
    return is_good_for_eqtl(sra_record) and bool(geo_record.get('gse'))

def estimate_file_size_gb(sra_record: Dict[str, Any]) -> float:
    """Estimate file size in GB."""
    bases = safe_int_convert(sra_record.get('bases', 0))
    if bases > 0:
        # Rough estimate: 1 base = 1 byte for FASTQ, compressed ~4x
        estimated_bytes = bases / 4
        return round(estimated_bytes / (1024 ** 3), 2)
    return 0.0

def generate_aspera_command(run_accession: str) -> str:
    """Generate Aspera download command."""
    if not run_accession:
        return ""
    
    prefix = run_accession[:6]
    return (f"ascp -QT -l 300m -P33001 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh "
            f"era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/{prefix}/{run_accession}/ ./")

if __name__ == "__main__":
    # Test the fixed function
    sample_sra = [{"run_accession": "SRR123", "study_title": "Test study", "scientific_name": "Homo sapiens"}]
    sample_geo = [{"gse": "GSE123", "gse_title": "Test GEO", "organism": "Homo sapiens"}]
    mapping = {"geo_to_sra": {}}
    
    result = create_integrated_dataset_table_fixed(sample_geo, sample_sra, mapping)
    print(f"Created {len(result)} integrated records") 