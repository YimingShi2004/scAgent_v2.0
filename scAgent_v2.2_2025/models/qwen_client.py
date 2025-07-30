"""
Qwen model client for scAgent.
"""

import requests
import json
from typing import Dict, Any, Optional, List
import logging
from dynaconf import Dynaconf

from .base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

# Load configuration
import os
from pathlib import Path

# Get the directory where this module is located
current_dir = Path(__file__).parent.parent
settings_file = current_dir / "settings.yml"

settings = Dynaconf(
    envvar_prefix="SCAGENT",
    settings_files=[str(settings_file), ".secrets.yml"],
    environments=True,
    load_dotenv=True,
)

class QwenClient(BaseModel):
    """Client for Qwen model API."""
    
    def __init__(
        self,
        model_name: str = "Qwen3-235B-A22B",
        api_base: str = "http://10.28.1.21:30080/v1",
        **kwargs
    ):
        super().__init__(model_name, api_base, **kwargs)
        self.api_url = f"{api_base}/chat/completions"
        self.headers = {
            "Content-Type": "application/json"
        }
        
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the model."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 32000,
        **kwargs
    ) -> ModelResponse:
        """Have a chat conversation with the model."""
        
        # Prepare request parameters
        request_params = self._prepare_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Add Qwen-specific parameters
        if "enable_thinking" in kwargs:
            if not kwargs["enable_thinking"]:
                # No-thinking mode
                request_params.update({
                    "top_p": kwargs.get("top_p", 0.8),
                    "top_k": kwargs.get("top_k", 20),
                    "presence_penalty": kwargs.get("presence_penalty", 1.5),
                    "chat_template_kwargs": {"enable_thinking": False}
                })
            else:
                # Thinking mode (default)
                request_params.update({
                    "top_p": kwargs.get("top_p", 0.95),
                    "top_k": kwargs.get("top_k", 20),
                })
        else:
            # Default to thinking mode
            request_params.update({
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 20),
            })
        
        try:
            logger.debug(f"Sending request to {self.api_url}")
            logger.debug(f"Request params: {json.dumps(request_params, indent=2)}")
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_params,
                timeout=60
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            logger.debug(f"Response: {json.dumps(response_data, indent=2)}")
            
            return self._handle_response(response_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return ModelResponse(
                content=f"Error: Request failed - {str(e)}",
                raw_response={"error": str(e)}
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response JSON: {e}")
            return ModelResponse(
                content="Error: Failed to parse response",
                raw_response={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ModelResponse(
                content=f"Error: {str(e)}",
                raw_response={"error": str(e)}
            )
    
    def chat_thinking(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 32000,
        **kwargs
    ) -> ModelResponse:
        """Chat with thinking mode enabled."""
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=True,
            top_p=0.95,
            top_k=20,
            **kwargs
        )
    
    def chat_no_thinking(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 32000,
        **kwargs
    ) -> ModelResponse:
        """Chat with thinking mode disabled."""
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=False,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            **kwargs
        )
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the model API."""
        try:
            test_response = self.generate(
                "Hello, this is a test message. Please respond with 'Test successful'.",
                temperature=0.1,
                max_tokens=100
            )
            
            return {
                "status": "success",
                "api_url": self.api_url,
                "model": self.model_name,
                "response": test_response.content,
                "usage": test_response.usage
            }
        except Exception as e:
            return {
                "status": "failed",
                "api_url": self.api_url,
                "model": self.model_name,
                "error": str(e)
            }

def get_qwen_client(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> QwenClient:
    """Get a Qwen client with configuration from settings."""
    
    # Use settings if not provided
    model_name = model_name or settings.model_name
    api_base = api_base or settings.model_api_base
    
    return QwenClient(
        model_name=model_name,
        api_base=api_base,
        **kwargs
    )

def create_analysis_prompt(
    task: str,
    data_info: str,
    context: Optional[str] = None
) -> str:
    """Create a structured prompt for data analysis tasks."""
    
    prompt = f"""You are a bioinformatics data analyst specializing in single-cell RNA-seq data for eQTL analysis.

Task: {task}

Data Information:
{data_info}
"""
    
    if context:
        prompt += f"\nContext:\n{context}"
    
    prompt += """

For sc-eQTL analysis, please evaluate the data based on these critical criteria:

**Essential Requirements:**
1. **Genotype Data Availability**: sc-eQTL requires paired genotype and transcriptome data from the same individuals
2. **Multi-individual Design**: Need multiple donors/individuals (minimum 20-50 for adequate power)
3. **Sufficient Cell Numbers**: At least 50-100 cells per individual for reliable expression estimates
4. **Quality Metrics**: Sequencing depth, gene detection rates, mitochondrial content
5. **Metadata Completeness**: Individual IDs, cell types, tissue information, batch information

**Assessment Framework:**
- **High Priority**: Datasets with confirmed genotype linkage and multi-individual design
- **Medium Priority**: Large cell numbers but unclear genotype availability
- **Low Priority**: Single individual studies or missing critical metadata

**Technical Considerations:**
- Platform compatibility (10X, Smart-seq2, etc.)
- Batch effects between sequencing runs
- Cell type heterogeneity and annotation quality
- Tissue-specific expression patterns

Please provide:
1. **Summary of Findings**: Overview of dataset characteristics
2. **Relevance for sc-eQTL Analysis**: Specific assessment of eQTL suitability with scoring
3. **Data Quality Assessment**: Technical quality metrics and concerns
4. **Recommendations for Filtering/Processing**: Specific steps to improve dataset quality
5. **Potential Issues or Limitations**: Critical problems that may prevent sc-eQTL analysis

Format your response in a clear, structured manner with specific recommendations and prioritization."""
    
    return prompt

def create_eqtl_evaluation_prompt(
    datasets: List[Dict[str, Any]],
    search_criteria: Dict[str, Any]
) -> str:
    """Create a specialized prompt for sc-eQTL dataset evaluation."""
    
    prompt = f"""You are a computational biologist specializing in single-cell eQTL (sc-eQTL) analysis.

TASK: Evaluate {len(datasets)} datasets for sc-eQTL analysis potential.

SEARCH CRITERIA:
{json.dumps(search_criteria, indent=2)}

DATASETS TO EVALUATE:
{json.dumps(datasets, indent=2, default=str)}

EVALUATION FRAMEWORK FOR SC-EQTL:

**Critical Requirements (Must Have):**
1. **Genotype Data**: Paired SNP/genotype data for each individual
2. **Multi-individual Design**: ≥20 individuals for adequate statistical power
3. **Cell-level Expression**: Single-cell resolution transcriptome data
4. **Individual Mapping**: Clear mapping between cells and donor individuals

**Quality Indicators (Should Have):**
1. **Sample Size**: ≥100 cells per individual for robust expression estimates
2. **Gene Coverage**: ≥15,000 detected genes per dataset
3. **Sequencing Depth**: ≥1,000 UMIs per cell (for droplet-based methods)
4. **Metadata Quality**: Complete cell type, tissue, and batch annotations

**Technical Considerations:**
1. **Platform Compatibility**: 10X Genomics, Smart-seq2, etc.
2. **Batch Effects**: Multiple sequencing runs may require correction
3. **Cell Type Diversity**: Sufficient cells per cell type for type-specific eQTL
4. **Tissue Specificity**: Relevant tissues for eQTL discovery

**Evaluation Scoring:**
- **A-Grade (Excellent)**: Meets all critical requirements + high quality indicators
- **B-Grade (Good)**: Meets critical requirements + some quality indicators
- **C-Grade (Marginal)**: Missing some critical requirements but potentially usable
- **D-Grade (Poor)**: Missing multiple critical requirements

PLEASE PROVIDE:

### **1. Summary of Findings**
- Dataset overview with key characteristics
- Distribution of sample sizes, organisms, and tissues

### **2. Relevance for sc-eQTL Analysis**
- Grade each dataset (A/B/C/D) with specific justification
- Identify datasets with highest eQTL potential
- Note any datasets with confirmed genotype linkage

### **3. Data Quality Assessment**
- Technical quality evaluation (sequencing platforms, depth, etc.)
- Metadata completeness and accuracy
- Potential batch effects or technical issues

### **4. Recommendations for Filtering/Processing**
- Priority ranking of datasets for eQTL analysis
- Required preprocessing steps (batch correction, QC, etc.)
- Suggested filtering criteria for cell and gene selection

### **5. Potential Issues or Limitations**
- Critical missing information (especially genotype data)
- Technical limitations that may affect eQTL detection
- Sample size or power considerations

**IMPORTANT**: If genotype data availability is not explicitly mentioned in the metadata, clearly flag this as a critical limitation that must be verified before proceeding with eQTL analysis."""
    
    return prompt

def create_data_cleaning_prompt(
    table_name: str,
    data_summary: Dict[str, Any],
    target_analysis: str = "sc-eQTL"
) -> str:
    """Create a specialized prompt for data cleaning and filtering."""
    
    prompt = f"""You are a data curation specialist for single-cell genomics data.

TASK: Clean and filter {table_name} data for {target_analysis} analysis.

DATA SUMMARY:
{json.dumps(data_summary, indent=2, default=str)}

CLEANING OBJECTIVES FOR {target_analysis.upper()}:

**Data Quality Issues to Address:**
1. **Metadata Standardization**: Inconsistent organism names, tissue terms, cell types
2. **Missing Critical Information**: Individual IDs, genotype data availability
3. **Technical Annotations**: Platform inconsistencies, library preparation methods
4. **Sample Size Validation**: Verify cell counts and individual numbers

**Filtering Criteria:**
1. **Organism Filter**: Focus on model organisms (Homo sapiens, Mus musculus)
2. **Technology Filter**: Prioritize established sc-RNA platforms
3. **Sample Size Filter**: Minimum thresholds for cells and individuals
4. **Metadata Completeness**: Require essential annotation fields

**Standardization Tasks:**
1. **Organism Names**: Standardize to scientific names
2. **Tissue Terms**: Map to standard anatomical ontologies
3. **Cell Type Terms**: Align with Cell Ontology (CL) terms
4. **Platform Names**: Standardize sequencing platform descriptions

PLEASE PROVIDE:

### **1. Data Quality Assessment**
- Overview of data quality issues identified
- Completeness metrics for critical fields
- Inconsistencies in metadata annotations

### **2. Recommended Cleaning Steps**
- Specific standardization procedures
- Missing data imputation strategies
- Quality control thresholds

### **3. Filtering Criteria for {target_analysis.upper()} Suitability**
- Organism-specific filters
- Technology and platform requirements
- Sample size and quality thresholds

### **4. Identification of Missing or Problematic Data**
- Records with insufficient metadata
- Potential annotation errors
- Critical missing information

### **5. Suggestions for Data Standardization**
- Controlled vocabulary mapping
- Ontology alignment procedures
- Batch processing recommendations

Focus on maximizing data utility for {target_analysis} analysis while maintaining high quality standards."""
    
    return prompt 