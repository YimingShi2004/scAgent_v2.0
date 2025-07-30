#!/usr/bin/env python3
"""
Improved assessment functions for better identification of clear pass/fail cases
"""

from typing import Dict, List, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)

def assess_species_improved(record: Dict[str, Any], required_species: List[str]) -> Dict[str, Any]:
    """
    Improved species assessment with better identification of clear cases.
    """
    # Get all organism-related fields
    organism_fields = [
        record.get('organism', ''),
        record.get('scientific_name', ''),
        record.get('sra_organism', ''),
        record.get('geo_organism', ''),
        record.get('organism_ch1', ''),
        record.get('common_name', ''),
        str(record.get('taxon_id', ''))
    ]
    
    # Combine all text for analysis
    combined_text = ' '.join(str(field).lower() for field in organism_fields if field)
    
    # Clear positive indicators
    clear_positive_patterns = [
        r'\bhomo sapiens\b',
        r'\bhuman\b(?!\s+(?:cell|line))',  # Human but not "human cell line"
        r'\btaxon[_\s]*id[:\s]*9606\b',
        r'\bhomo[_\s]+sapiens\b'
    ]
    
    # Clear negative indicators  
    clear_negative_patterns = [
        r'\bmouse\b', r'\bmus musculus\b',
        r'\brat\b', r'\brattus\b',
        r'\bzebrafish\b', r'\bdanio rerio\b',
        r'\byeast\b', r'\bsaccharomyces\b',
        r'\bfly\b', r'\bdrosophila\b',
        r'\bworm\b', r'\bc\.?\s*elegans\b'
    ]
    
    # Check for clear positive matches
    for pattern in clear_positive_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return {
                'score': 3,
                'confidence': 0.95,
                'decision': 'accept',
                'reasons': [f'Clear human species identification: {pattern}'],
                'evidence': combined_text[:100]
            }
    
    # Check for clear negative matches
    for pattern in clear_negative_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return {
                'score': 0,
                'confidence': 0.90,
                'decision': 'reject',
                'reasons': [f'Non-human species detected: {pattern}'],
                'evidence': combined_text[:100]
            }
    
    # Check for ambiguous cases that need AI
    ambiguous_patterns = [
        r'\bcell\s+line\b',
        r'\bprimary\s+cell\b',
        r'\btissue\b',
        r'\bsample\b'
    ]
    
    has_ambiguous = any(re.search(pattern, combined_text, re.IGNORECASE) 
                       for pattern in ambiguous_patterns)
    
    if has_ambiguous and combined_text:
        return {
            'score': 1,  # Neutral score
            'confidence': 0.3,  # Low confidence - needs AI
            'decision': 'uncertain',
            'reasons': ['Species information ambiguous, needs AI review'],
            'evidence': combined_text[:100]
        }
    
    # No clear information
    return {
        'score': 0,
        'confidence': 0.8,
        'decision': 'reject',
        'reasons': ['No species information found'],
        'evidence': combined_text[:100]
    }

def assess_database_id_improved(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improved database ID assessment.
    """
    # Database ID patterns
    id_patterns = {
        'SRA_RUN': r'^SRR\d{6,}$',
        'SRA_STUDY': r'^SRP\d{6,}$',
        'GEO_SERIES': r'^GSE\d{4,}$',
        'GEO_SAMPLE': r'^GSM\d{6,}$',
        'DBGAP': r'^phs\d{6}',
        'PMID': r'^\d{7,8}$'
    }
    
    # Get all ID-related fields
    id_fields = [
        record.get('run_accession', ''),
        record.get('sra_run_accession', ''),
        record.get('study_accession', ''),
        record.get('sra_study_accession', ''),
        record.get('geo_accession', ''),
        record.get('gse', ''),
        record.get('gsm', ''),
        record.get('pmid', ''),
        record.get('dbgap_id', '')
    ]
    
    valid_ids = []
    id_types = []
    
    for field_value in id_fields:
        if not field_value:
            continue
            
        field_str = str(field_value).strip()
        if len(field_str) < 4:  # Too short to be valid
            continue
            
        for id_type, pattern in id_patterns.items():
            if re.match(pattern, field_str, re.IGNORECASE):
                valid_ids.append(field_str)
                id_types.append(id_type)
                break
    
    if valid_ids:
        return {
            'score': 3,
            'confidence': 0.95,
            'decision': 'accept',
            'reasons': [f'Valid database IDs found: {", ".join(id_types)}'],
            'evidence': ', '.join(valid_ids)
        }
    
    # Check for potential IDs that might be valid but don't match patterns
    potential_ids = [f for f in id_fields if f and len(str(f).strip()) >= 6]
    
    if potential_ids:
        return {
            'score': 1,
            'confidence': 0.4,
            'decision': 'uncertain',
            'reasons': ['Potential database IDs found but format unclear'],
            'evidence': ', '.join(str(id) for id in potential_ids[:3])
        }
    
    return {
        'score': 0,
        'confidence': 0.95,
        'decision': 'reject',
        'reasons': ['No valid database IDs found'],
        'evidence': 'No IDs'
    }

def assess_cell_line_improved(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improved cell line assessment to better identify clear cases.
    """
    # Get text fields for analysis
    text_fields = [
        record.get('study_title', ''),
        record.get('geo_title', ''),
        record.get('sra_study_title', ''),
        record.get('summary', ''),
        record.get('geo_summary', ''),
        record.get('study_abstract', ''),
        record.get('sample_name', ''),
        record.get('source_name', ''),
        record.get('characteristics', ''),
        record.get('description', '')
    ]
    
    combined_text = ' '.join((str(field) or '').lower() for field in text_fields)
    
    # Clear cell line indicators (high confidence rejection)
    clear_cell_line_patterns = [
        r'\bcell\s+line\b',
        r'\bhela\b', r'\bk562\b', r'\bjurkat\b', r'\b293t?\b', r'\bu937\b',
        r'\bthp-?1\b', r'\bmcf-?7\b', r'\ba549\b', r'\bcos-?7\b',
        r'\bvero\b', r'\bmdck\b', r'\bcho\b', r'\bnih-?3t3\b'
    ]
    
    # Clear primary tissue/cell indicators (accept)
    clear_primary_patterns = [
        r'\bprimary\s+(?:cell|tissue|culture)\b',
        r'\bfresh\s+tissue\b',
        r'\bbiopsy\b',
        r'\bpatient\s+(?:sample|tissue|cell)\b',
        r'\bdonor\s+(?:sample|tissue|cell)\b',
        r'\bin\s+vivo\b',
        r'\bsurgical\s+specimen\b'
    ]
    
    # Check for clear cell line indicators
    for pattern in clear_cell_line_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            return {
                'score': 0,  # Reject cell lines
                'confidence': 0.90,
                'decision': 'reject',
                'reasons': [f'Cell line detected: {match.group()}'],
                'evidence': combined_text[:100]
            }
    
    # Check for clear primary tissue indicators
    for pattern in clear_primary_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            return {
                'score': 3,  # Accept primary tissues
                'confidence': 0.85,
                'decision': 'accept',
                'reasons': [f'Primary tissue/cell detected: {match.group()}'],
                'evidence': combined_text[:100]
            }
    
    # Ambiguous cases that need AI review
    ambiguous_patterns = [
        r'\bcell\b(?!\s+line)',  # "cell" but not "cell line"
        r'\bculture\b',
        r'\bisolated\b',
        r'\bsorted\b'
    ]
    
    has_ambiguous = any(re.search(pattern, combined_text, re.IGNORECASE) 
                       for pattern in ambiguous_patterns)
    
    if has_ambiguous and combined_text:
        return {
            'score': 2,  # Neutral-positive score
            'confidence': 0.4,  # Low confidence - needs AI
            'decision': 'uncertain',
            'reasons': ['Cell type information ambiguous, needs AI review'],
            'evidence': combined_text[:100]
        }
    
    # Default: assume primary tissue if no clear indicators
    return {
        'score': 2,
        'confidence': 0.6,
        'decision': 'accept',
        'reasons': ['No clear cell line indicators found'],
        'evidence': 'No clear indicators'
    }

def assess_sequencing_method_improved(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improved sequencing method assessment.
    """
    # Get method-related fields
    method_fields = [
        record.get('library_strategy', ''),
        record.get('sra_library_strategy', ''),
        record.get('library_selection', ''),
        record.get('library_source', ''),
        record.get('platform', ''),
        record.get('sra_platform', ''),
        record.get('instrument_model', ''),
        record.get('study_title', ''),
        record.get('geo_title', '')
    ]
    
    combined_text = ' '.join((str(field) or '').lower() for field in method_fields)
    
    # Clear single-cell method indicators
    clear_sc_patterns = [
        r'\bsingle[_\s]*cell\b',
        r'\bsc[_\s]*rna[_\s]*seq\b',
        r'\b10x\s*genomics?\b',
        r'\bdrop[_\s]*seq\b',
        r'\bsmart[_\s]*seq\b',
        r'\bcel[_\s]*seq\b',
        r'\bmars[_\s]*seq\b',
        r'\bscrb[_\s]*seq\b'
    ]
    
    # Clear bulk RNA-seq indicators (still acceptable but lower score)
    bulk_rna_patterns = [
        r'\brna[_\s]*seq\b(?!\s*single)',
        r'\btranscriptom\w*\b'
    ]
    
    # Check for clear single-cell methods
    for pattern in clear_sc_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            return {
                'score': 3,
                'confidence': 0.90,
                'decision': 'accept',
                'reasons': [f'Single-cell method detected: {match.group()}'],
                'evidence': combined_text[:100]
            }
    
    # Check for bulk RNA-seq
    for pattern in bulk_rna_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            return {
                'score': 2,
                'confidence': 0.75,
                'decision': 'accept',
                'reasons': [f'RNA-seq method detected: {match.group()}'],
                'evidence': combined_text[:100]
            }
    
    # Check for other sequencing methods
    other_seq_patterns = [
        r'\bchip[_\s]*seq\b',
        r'\batac[_\s]*seq\b',
        r'\bwgs\b', r'\bwhole\s+genome\b',
        r'\bexome\b'
    ]
    
    for pattern in other_seq_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            return {
                'score': 1,
                'confidence': 0.80,
                'decision': 'neutral',
                'reasons': [f'Other sequencing method: {match.group()}'],
                'evidence': combined_text[:100]
            }
    
    # Uncertain cases
    if combined_text and ('illumina' in combined_text or 'sequencing' in combined_text):
        return {
            'score': 1,
            'confidence': 0.3,
            'decision': 'uncertain',
            'reasons': ['Sequencing method unclear, needs AI review'],
            'evidence': combined_text[:100]
        }
    
    return {
        'score': 0,
        'confidence': 0.7,
        'decision': 'reject',
        'reasons': ['No clear sequencing method found'],
        'evidence': 'No method info'
    }

# Function to patch the existing assessment functions
def patch_assessment_functions():
    """Replace existing assessment functions with improved versions."""
    import scAgent.utils as utils_module
    
    # Replace the assessment functions
    utils_module.assess_species_with_confidence = lambda record, species_list: assess_species_improved(record, species_list)
    utils_module.assess_database_id_with_confidence = lambda record: assess_database_id_improved(record)
    utils_module.assess_cell_line_with_confidence = lambda record: assess_cell_line_improved(record)
    utils_module.assess_sequencing_method_with_confidence = lambda record: assess_sequencing_method_improved(record)
    
    logger.info("Assessment functions patched with improved versions")

if __name__ == "__main__":
    # Test improved assessments
    test_record = {
        'organism': 'Homo sapiens',
        'run_accession': 'SRR12345678',
        'study_title': 'Single cell RNA-seq of human brain tissue',
        'library_strategy': 'RNA-Seq'
    }
    
    print("Testing improved assessments:")
    print("Species:", assess_species_improved(test_record, ['Homo sapiens']))
    print("Database ID:", assess_database_id_improved(test_record))
    print("Cell line:", assess_cell_line_improved(test_record))
    print("Sequencing method:", assess_sequencing_method_improved(test_record)) 