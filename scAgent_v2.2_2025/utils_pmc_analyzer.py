#!/usr/bin/env python3
"""
PMC PDF Analyzer for Enhanced sc-eQTL Dataset Discovery
Extracts detailed information from PMC PDF documents including:
- Age information
- Sample size details
- Geographic location
- Disease status
- Publication metadata
- Experimental protocols
"""

import logging
import re
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
from urllib.parse import urljoin, quote
import xml.etree.ElementTree as ET

from .models.client import get_model_client

logger = logging.getLogger(__name__)

class PmcAnalyzer:
    """
    PMC PDF analyzer for extracting detailed information from scientific publications.
    """
    
    def __init__(self, ai_client: Optional[Any] = None):
        self.ai_client = ai_client or get_model_client()
        self.pmc_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.pmc_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.pmc_summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        
        # Age patterns for extraction from PMC documents
        self.age_patterns = [
            r'age\s*:\s*([^;,\n]+)',
            r'age\s*=\s*([^;,\n]+)',
            r'(\d+)\s*(?:years?|y\.?o\.?|yrs?)',
            r'(\d+)\s*(?:months?|mos?)',
            r'(\d+)\s*(?:days?|d)',
            r'p(\d+)',  # Postnatal days
            r'embryonic\s*day\s*(\d+)',
            r'e(\d+)',  # Embryonic day shorthand
            r'(\d+)\s*(?:week|wk)s?',
            r'(\d+)\s*(?:hour|hr)s?',
            r'age\s*range\s*:\s*([^;,\n]+)',
            r'aged\s*(\d+)\s*(?:years?|y\.?o\.?|yrs?)',
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|y\.?o\.?|yrs?)',  # Age ranges
        ]
        
        # Sample size patterns
        self.sample_size_patterns = [
            r'(\d+)\s*(?:samples?|subjects?|patients?|individuals?|donors?|participants?)',
            r'n\s*=\s*(\d+)',
            r'(\d+)\s*(?:single.?cells?|cells?)',
            r'from\s*(\d+)\s*(?:subjects?|patients?|individuals?)',
            r'total\s*of\s*(\d+)\s*(?:samples?|subjects?|patients?)',
            r'(\d+)\s*(?:males?|females?)',
            r'(\d+)\s*(?:cases?|controls?)',
            r'(\d+)\s*(?:tissues?|organs?)',
        ]
        
        # Geographic patterns
        self.geographic_patterns = [
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:university|hospital|institute|center)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:medical|research|clinical)',
        ]
        
        # Disease patterns
        self.disease_patterns = [
            r'(?:patients?|subjects?)\s+with\s+([^;,\n]+)',
            r'(?:diagnosed|confirmed)\s+with\s+([^;,\n]+)',
            r'([^;,\n]+)\s+(?:patients?|subjects?|cases?)',
            r'(?:healthy|normal|control)\s+(?:subjects?|individuals?|donors?)',
        ]
    
    def search_pmc_by_pmid(self, pmid: str) -> Optional[str]:
        """
        Search PMC for a given PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PMC ID if found, None otherwise
        """
        try:
            # Search for PMC ID using PMID
            search_params = {
                'db': 'pmc',
                'term': f'{pmid}[pmid]',
                'retmode': 'json',
                'retmax': 1
            }
            
            response = requests.get(self.pmc_search_url, params=search_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                pmc_ids = data['esearchresult']['idlist']
                if pmc_ids:
                    return pmc_ids[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to search PMC for PMID {pmid}: {e}")
            return None
    
    def get_pmc_summary(self, pmc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get PMC article summary.
        
        Args:
            pmc_id: PMC ID
            
        Returns:
            Article summary dictionary
        """
        try:
            summary_params = {
                'db': 'pmc',
                'id': pmc_id,
                'retmode': 'json',
                'rettype': 'abstract'
            }
            
            response = requests.get(self.pmc_summary_url, params=summary_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'result' in data and pmc_id in data['result']:
                return data['result'][pmc_id]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get PMC summary for {pmc_id}: {e}")
            return None
    
    def get_pmc_full_text(self, pmc_id: str) -> Optional[str]:
        """
        Get full text content from PMC.
        
        Args:
            pmc_id: PMC ID
            
        Returns:
            Full text content
        """
        try:
            fetch_params = {
                'db': 'pmc',
                'id': pmc_id,
                'retmode': 'xml',
                'rettype': 'full'
            }
            
            response = requests.get(self.pmc_fetch_url, params=fetch_params, timeout=60)
            response.raise_for_status()
            
            # Parse XML to extract text content
            root = ET.fromstring(response.content)
            
            # Extract text from various sections
            text_sections = []
            
            # Abstract
            abstracts = root.findall('.//abstract')
            for abstract in abstracts:
                text_sections.append(ET.tostring(abstract, encoding='unicode', method='text'))
            
            # Methods
            methods = root.findall('.//sec[@sec-type="methods"]')
            for method in methods:
                text_sections.append(ET.tostring(method, encoding='unicode', method='text'))
            
            # Results
            results = root.findall('.//sec[@sec-type="results"]')
            for result in results:
                text_sections.append(ET.tostring(result, encoding='unicode', method='text'))
            
            # Discussion
            discussions = root.findall('.//sec[@sec-type="discussion"]')
            for discussion in discussions:
                text_sections.append(ET.tostring(discussion, encoding='unicode', method='text'))
            
            # Combine all text
            full_text = ' '.join(text_sections)
            
            # Clean up text
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            return full_text if full_text else None
            
        except Exception as e:
            logger.warning(f"Failed to get PMC full text for {pmc_id}: {e}")
            return None
    
    def extract_age_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract age information from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Age information dictionary
        """
        age_info = {
            "age_value": "",
            "age_unit": "",
            "age_range": "",
            "confidence": 0.0,
            "source_text": ""
        }
        
        text_lower = text.lower()
        
        for pattern in self.age_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                if len(matches[0]) == 2 and isinstance(matches[0], tuple):  # Age range
                    age_info.update({
                        "age_range": f"{matches[0][0]}-{matches[0][1]}",
                        "age_unit": "years",
                        "confidence": 0.8,
                        "source_text": matches[0]
                    })
                else:
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
                        "confidence": 0.9,
                        "source_text": age_value
                    })
                break
        
        return age_info
    
    def extract_sample_size_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract sample size information from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Sample size information dictionary
        """
        sample_info = {
            "sample_size": "",
            "sample_type": "",
            "confidence": 0.0,
            "source_text": ""
        }
        
        text_lower = text.lower()
        
        for pattern in self.sample_size_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Find the largest number
                max_size = 0
                sample_type = ""
                
                for match in matches:
                    if isinstance(match, str) and match.isdigit():
                        size = int(match)
                        if size > max_size:
                            max_size = size
                            sample_type = pattern
                
                if max_size > 0:
                    sample_info.update({
                        "sample_size": str(max_size),
                        "sample_type": sample_type,
                        "confidence": 0.8,
                        "source_text": str(max_size)
                    })
                    break
        
        return sample_info
    
    def extract_geographic_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract geographic information from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Geographic information dictionary
        """
        geo_info = {
            "location": "",
            "institution": "",
            "confidence": 0.0,
            "source_text": ""
        }
        
        text_lower = text.lower()
        
        for pattern in self.geographic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                location = matches[0].strip()
                geo_info.update({
                    "location": location,
                    "confidence": 0.7,
                    "source_text": location
                })
                break
        
        return geo_info
    
    def extract_disease_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract disease information from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Disease information dictionary
        """
        disease_info = {
            "disease_status": "",
            "disease_type": "",
            "confidence": 0.0,
            "source_text": ""
        }
        
        text_lower = text.lower()
        
        # Check for healthy/control mentions
        if any(term in text_lower for term in ['healthy', 'normal', 'control']):
            disease_info.update({
                "disease_status": "healthy",
                "disease_type": "control",
                "confidence": 0.8,
                "source_text": "healthy/normal/control"
            })
        else:
            # Check for disease patterns
            for pattern in self.disease_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    disease = matches[0].strip()
                    disease_info.update({
                        "disease_status": "disease",
                        "disease_type": disease,
                        "confidence": 0.7,
                        "source_text": disease
                    })
                    break
        
        return disease_info
    
    def analyze_pmc_with_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI to analyze PMC text for detailed information.
        
        Args:
            text: PMC text content
            
        Returns:
            AI analysis results
        """
        try:
            prompt = f"""
            Analyze this scientific publication text and extract detailed information for sc-eQTL analysis:

            Text: {text[:2000]}

            Please extract and return ONLY the following information in JSON format:
            {{
                "age_information": {{
                    "age_value": "Specific age or age range if mentioned",
                    "age_unit": "years/months/days/weeks/hours",
                    "age_range": "Age range if specified",
                    "confidence": 0.0-1.0
                }},
                "sample_size": {{
                    "total_samples": "Total number of samples/subjects",
                    "sample_type": "Type of samples (cells, tissues, individuals, etc.)",
                    "confidence": 0.0-1.0
                }},
                "geographic_location": {{
                    "country": "Country if mentioned",
                    "institution": "Institution or hospital if mentioned",
                    "confidence": 0.0-1.0
                }},
                "disease_status": {{
                    "status": "healthy/disease/unknown",
                    "disease_type": "Specific disease if mentioned",
                    "confidence": 0.0-1.0
                }},
                "experimental_details": {{
                    "tissue_type": "Type of tissue analyzed",
                    "cell_type": "Type of cells analyzed",
                    "sequencing_platform": "Sequencing platform used",
                    "confidence": 0.0-1.0
                }}
            }}

            Use "Not specified" if information is not available.
            Provide confidence scores based on how clearly the information is stated.
            """
            
            response = self.ai_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI response as JSON")
                return self._extract_fallback_info(text)
                
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return self._extract_fallback_info(text)
    
    def _extract_fallback_info(self, text: str) -> Dict[str, Any]:
        """
        Fallback extraction when AI analysis fails.
        
        Args:
            text: Text content
            
        Returns:
            Fallback extraction results
        """
        return {
            "age_information": self.extract_age_from_text(text),
            "sample_size": self.extract_sample_size_from_text(text),
            "geographic_location": self.extract_geographic_from_text(text),
            "disease_status": self.extract_disease_from_text(text),
            "experimental_details": {
                "tissue_type": "Not specified",
                "cell_type": "Not specified", 
                "sequencing_platform": "Not specified",
                "confidence": 0.0
            }
        }
    
    def analyze_pmid_comprehensive(self, pmid: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a PMID through PMC.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Comprehensive analysis results
        """
        analysis_result = {
            "pmid": pmid,
            "pmc_id": None,
            "analysis_success": False,
            "extracted_info": {},
            "error_message": ""
        }
        
        try:
            # Step 1: Search for PMC ID
            pmc_id = self.search_pmc_by_pmid(pmid)
            if not pmc_id:
                analysis_result["error_message"] = "No PMC ID found for this PMID"
                return analysis_result
            
            analysis_result["pmc_id"] = pmc_id
            
            # Step 2: Get PMC summary
            summary = self.get_pmc_summary(pmc_id)
            if summary:
                analysis_result["extracted_info"]["summary"] = summary
            
            # Step 3: Get full text
            full_text = self.get_pmc_full_text(pmc_id)
            if not full_text:
                analysis_result["error_message"] = "Could not retrieve full text from PMC"
                return analysis_result
            
            # Step 4: Analyze with AI
            ai_analysis = self.analyze_pmc_with_ai(full_text)
            analysis_result["extracted_info"].update(ai_analysis)
            
            # Step 5: Extract additional metadata
            analysis_result["extracted_info"]["publication_metadata"] = {
                "title": summary.get("title", "Not available") if summary else "Not available",
                "authors": summary.get("authors", "Not available") if summary else "Not available",
                "journal": summary.get("journal", "Not available") if summary else "Not available",
                "publication_date": summary.get("pubdate", "Not available") if summary else "Not available",
                "doi": summary.get("elocationid", "Not available") if summary else "Not available"
            }
            
            analysis_result["analysis_success"] = True
            
        except Exception as e:
            analysis_result["error_message"] = str(e)
            logger.error(f"Comprehensive PMC analysis failed for PMID {pmid}: {e}")
        
        return analysis_result
    
    def batch_analyze_pmids(self, pmids: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Batch analyze multiple PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of analysis results
        """
        results = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pmid = {
                executor.submit(self.analyze_pmid_comprehensive, pmid): pmid 
                for pmid in pmids
            }
            
            for future in as_completed(future_to_pmid):
                result = future.result()
                results.append(result)
                
                # Add delay to respect API rate limits
                time.sleep(0.1)
        
        return results 