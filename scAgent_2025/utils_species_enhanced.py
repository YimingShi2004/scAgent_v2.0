#!/usr/bin/env python3
"""
Enhanced species detection for better human record identification
"""

from typing import Dict, List, Any
import re
import logging

logger = logging.getLogger(__name__)

def assess_species_enhanced(record: Dict[str, Any], required_species: List[str]) -> Dict[str, Any]:
    """
    Enhanced species assessment with comprehensive human detection.
    """
    # Get ALL possible organism-related fields
    organism_fields = [
        record.get('organism', ''),
        record.get('scientific_name', ''),
        record.get('sra_organism', ''),
        record.get('geo_organism', ''),
        record.get('organism_ch1', ''),
        record.get('common_name', ''),
        str(record.get('taxon_id', '')),
        # Additional fields that might contain organism info
        record.get('source_name', ''),
        record.get('source_name_ch1', ''),
        record.get('characteristics', ''),
        record.get('characteristics_ch1', ''),
        record.get('sample_name', ''),
        record.get('description', '')
    ]
    
    # Get ALL text fields that might contain species info
    text_fields = [
        record.get('study_title', ''),
        record.get('geo_title', ''),
        record.get('sra_study_title', ''),
        record.get('gse_title', ''),
        record.get('summary', ''),
        record.get('geo_summary', ''),
        record.get('study_abstract', ''),
        record.get('design_description', ''),
        record.get('library_source', ''),
        record.get('sample_title', ''),
        record.get('gsm_title', '')
    ]
    
    # Combine all fields for comprehensive analysis
    organism_text = ' '.join(str(field).lower().strip() for field in organism_fields if field)
    content_text = ' '.join(str(field).lower().strip() for field in text_fields if field)
    all_text = f"{organism_text} {content_text}"
    
    # 1. HIGHEST CONFIDENCE: Taxon ID check
    taxon_id = str(record.get('taxon_id', '')).strip()
    if taxon_id == '9606':
        return {
            'score': 3,
            'confidence': 1.0,
            'decision': 'accept',
            'reasons': ['Human taxon ID 9606 found'],
            'evidence': f'taxon_id={taxon_id}'
        }
    
    # 2. HIGH CONFIDENCE: Explicit scientific names
    explicit_human_patterns = [
        r'\bhomo\s+sapiens\b',
        r'\bh\.\s*sapiens\b',
        r'\bhsapiens\b',
        r'^human$',  # Exact match for "human" as organism
        r'homo sapiens sapiens'
    ]
    
    for pattern in explicit_human_patterns:
        match = re.search(pattern, organism_text, re.IGNORECASE)
        if match:
            return {
                'score': 3,
                'confidence': 0.95,
                'decision': 'accept',
                'reasons': [f'Explicit human species: {match.group()}'],
                'evidence': match.group()
            }
    
    # 3. HIGH CONFIDENCE: Non-human species (reject immediately)
    non_human_patterns = [
        r'\bmouse\b', r'\bmus\s+musculus\b', r'\bm\.\s*musculus\b',
        r'\brat\b', r'\brattus\b', r'\brattus\s+norvegicus\b',
        r'\bzebrafish\b', r'\bdanio\s+rerio\b', r'\bd\.\s*rerio\b',
        r'\byeast\b', r'\bsaccharomyces\b', r'\bs\.\s*cerevisiae\b',
        r'\bfly\b', r'\bdrosophila\b', r'\bd\.\s*melanogaster\b',
        r'\bworm\b', r'\bc\.\s*elegans\b', r'\bcaenorhabditis\b',
        r'\barabidopsis\b', r'\ba\.\s*thaliana\b',
        r'\bchicken\b', r'\bgallus\s+gallus\b',
        r'\bpig\b', r'\bsus\s+scrofa\b', r'\bswine\b',
        r'\bcow\b', r'\bbos\s+taurus\b', r'\bbovine\b'
    ]
    
    for pattern in non_human_patterns:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            return {
                'score': 0,
                'confidence': 0.90,
                'decision': 'reject',
                'reasons': [f'Non-human species detected: {match.group()}'],
                'evidence': match.group()
            }
    
    # 4. MEDIUM-HIGH CONFIDENCE: Strong human indicators in text
    strong_human_patterns = [
        r'\bhuman\s+(?:patient|subject|participant|individual|donor|sample|tissue|cell|blood|brain|liver|lung|heart|kidney)\b',
        r'\b(?:patient|subject|participant|clinical|medical)\s+(?:sample|tissue|blood|data)\b',
        r'\bhuman\s+(?:cancer|tumor|carcinoma|disease|condition)\b',
        r'\b(?:breast|lung|liver|brain|kidney|heart|prostate|ovarian|colon)\s+(?:cancer|tumor|carcinoma)\b',
        r'\bhuman\s+(?:pbmc|peripheral\s+blood)\b',
        r'\bclinical\s+(?:trial|study|sample|specimen)\b',
        r'\bpatient\s+(?:derived|sample|tissue|blood|biopsy)\b',
        r'\bhospital\b.*\b(?:sample|tissue|patient)\b',
        r'\bmedical\s+center\b',
        r'\bhuman\s+genome\b',
        r'\bhomo\s+sapiens\b'  # In text content
    ]
    
    for pattern in strong_human_patterns:
        match = re.search(pattern, content_text, re.IGNORECASE)
        if match:
            return {
                'score': 3,
                'confidence': 0.85,
                'decision': 'accept',
                'reasons': [f'Strong human indicator: {match.group()}'],
                'evidence': match.group()
            }
    
    # 5. MEDIUM CONFIDENCE: Moderate human indicators
    moderate_human_patterns = [
        r'\bhuman\b(?!\s+(?:cell\s+line|immortalized))',  # "human" but not cell line
        r'\bpatient\b',
        r'\bclinical\b',
        r'\bmedical\b',
        r'\bhospital\b',
        r'\bdonor\b',
        r'\bsubject\b',
        r'\bparticipant\b',
        r'\bbiopsy\b',
        r'\bsurgical\s+specimen\b',
        r'\bprimary\s+(?:tissue|cell|sample)\b'
    ]
    
    human_indicators_found = []
    for pattern in moderate_human_patterns:
        matches = re.findall(pattern, content_text, re.IGNORECASE)
        if matches:
            human_indicators_found.extend(matches)
    
    # If multiple moderate indicators, likely human
    if len(human_indicators_found) >= 3:
        return {
            'score': 2,
            'confidence': 0.75,
            'decision': 'accept',
            'reasons': [f'Multiple human indicators: {", ".join(set(human_indicators_found[:5]))}'],
            'evidence': f'Found {len(human_indicators_found)} indicators'
        }
    elif len(human_indicators_found) >= 2:
        return {
            'score': 2,
            'confidence': 0.65,
            'decision': 'accept',
            'reasons': [f'Some human indicators: {", ".join(set(human_indicators_found))}'],
            'evidence': f'Found {len(human_indicators_found)} indicators'
        }
    elif len(human_indicators_found) == 1:
        return {
            'score': 1,
            'confidence': 0.45,
            'decision': 'uncertain',
            'reasons': [f'Single human indicator: {human_indicators_found[0]}'],
            'evidence': human_indicators_found[0]
        }
    
    # 6. LOW CONFIDENCE: Disease/medical context (often human)
    disease_patterns = [
        r'\bcancer\b', r'\btumor\b', r'\btumour\b', r'\bcarcinoma\b',
        r'\badenocarcinoma\b', r'\bmalignant\b', r'\boncology\b',
        r'\bdisease\b', r'\bdisorder\b', r'\bsyndrome\b',
        r'\bdiagnosis\b', r'\btherapy\b', r'\btreatment\b'
    ]
    
    disease_matches = []
    for pattern in disease_patterns:
        matches = re.findall(pattern, content_text, re.IGNORECASE)
        if matches:
            disease_matches.extend(matches)
    
    if len(disease_matches) >= 2:
        return {
            'score': 1,
            'confidence': 0.55,
            'decision': 'uncertain',
            'reasons': [f'Disease context suggests human: {", ".join(set(disease_matches[:3]))}'],
            'evidence': f'Disease indicators: {len(disease_matches)}'
        }
    
    # 7. Check for any mention of "human" even if weak
    if re.search(r'\bhuman\b', all_text, re.IGNORECASE):
        return {
            'score': 1,
            'confidence': 0.40,
            'decision': 'uncertain',
            'reasons': ['Weak human mention found'],
            'evidence': 'Contains "human"'
        }
    
    # 8. No human indicators found
    return {
        'score': 0,
        'confidence': 0.70,
        'decision': 'reject',
        'reasons': ['No human species indicators found'],
        'evidence': f'Text length: {len(all_text)} chars'
    }

def patch_species_assessment():
    """Patch the species assessment with enhanced version."""
    import scAgent.utils as utils_module
    
    # Replace the species assessment function
    original_func = getattr(utils_module, 'assess_species_with_confidence', None)
    
    def enhanced_wrapper(record, species_list):
        return assess_species_enhanced(record, species_list)
    
    utils_module.assess_species_with_confidence = enhanced_wrapper
    logger.info("Species assessment patched with enhanced version")
    
    return original_func

if __name__ == "__main__":
    # Test enhanced species detection
    test_records = [
        {
            'organism': 'Homo sapiens',
            'study_title': 'Single cell RNA-seq of human brain tissue',
            'taxon_id': 9606
        },
        {
            'scientific_name': 'Mus musculus',
            'study_title': 'Mouse embryonic development'
        },
        {
            'study_title': 'Clinical study of breast cancer patients',
            'summary': 'We analyzed tumor samples from hospital patients'
        },
        {
            'study_title': 'Cell line experiment',
            'organism': 'human'
        }
    ]
    
    print("Testing enhanced species detection:")
    for i, record in enumerate(test_records, 1):
        result = assess_species_enhanced(record, ['Homo sapiens'])
        print(f"Record {i}: Score={result['score']}, Confidence={result['confidence']:.2f}, Decision={result['decision']}")
        print(f"  Reason: {result['reasons'][0]}")
        print(f"  Evidence: {result['evidence']}")
        print() 