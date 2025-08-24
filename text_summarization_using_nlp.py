from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, HttpUrl
from typing import Optional, Union, Annotated
import PyPDF2
import io
from transformers import pipeline
import uvicorn
import logging
import json
import re
import requests
from urllib.parse import urlparse
import time
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Text & PDF Summarizer API",
    description="An API that accepts text input, PDF files, or web URLs and returns intelligent summaries",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TextSummaryRequest(BaseModel):
    text: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Your long text content here that needs to be summarized. The summary length will be automatically calculated as 40% of the original text length for optimal readability and comprehension."
            }
        }
    }

class URLSummaryRequest(BaseModel):
    url: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4063875/"
            }
        }
    }

class SummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": "This is the generated summary of the input text.",
                "original_length": 1000,
                "summary_length": 100,
                "compression_ratio": 0.1
            }
        }
    }

class URLSummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    url: str
    title: str
    content_type: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": "This is the generated summary of the web content.",
                "original_length": 1000,
                "summary_length": 100,
                "compression_ratio": 0.1,
                "url": "https://example.com/article",
                "title": "Article Title",
                "content_type": "research_article"
            }
        }
    }

class PageWiseSummaryResponse(BaseModel):
    page_summaries: dict
    overall_summary: str
    total_pages: int
    content_pages: int
    total_original_length: int
    total_summary_length: int
    overall_compression_ratio: float
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "page_summaries": {
                    "1": {
                        "original_length": 2500,
                        "summary": "Page 1 summary content...",
                        "summary_length": 1000,
                        "compression_ratio": 0.4
                    }
                },
                "overall_summary": "Complete document summary combining all pages...",
                "total_pages": 15,
                "content_pages": 12,
                "total_original_length": 25000,
                "total_summary_length": 10000,
                "overall_compression_ratio": 0.4
            }
        }
    }

# Initialize summarization pipeline with proper configuration
try:
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        framework="pt"
    )
    logger.info("BART summarization model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load summarization model: {e}")
    summarizer = None

def detect_document_type(text: str) -> str:
    """Detect the type of document to determine summarization approach"""
    text_lower = text.lower()
    
    # Educational/instructional content
    if any(term in text_lower for term in ['lesson', 'chapter', 'module', 'objectives', 'learning', 'method', 'steps', 'process']):
        return "educational"
    
    # Technical/scientific content
    elif any(term in text_lower for term in ['research', 'study', 'analysis', 'methodology', 'results', 'conclusion', 'abstract']):
        return "technical"
    
    # Business/corporate content
    elif any(term in text_lower for term in ['company', 'business', 'market', 'revenue', 'strategy', 'management', 'report']):
        return "business"
    
    # Legal content
    elif any(term in text_lower for term in ['agreement', 'contract', 'legal', 'clause', 'terms', 'conditions', 'liability']):
        return "legal"
    
    # News/article content
    elif any(term in text_lower for term in ['breaking', 'reported', 'according to', 'sources', 'yesterday', 'today']):
        return "news"
    
    # General content
    else:
        return "general"

def extract_important_phrases(text: str) -> list:
    """Extract key multi-word phrases to preserve in summary"""
    
    # Common important phrase patterns
    phrase_patterns = [
        r'\b(?:machine learning|artificial intelligence|data science|deep learning)\b',
        r'\b(?:cost[- ]effective|time[- ]saving|energy[- ]efficient|cost[- ]benefit)\b',
        r'\b(?:real[- ]time|long[- ]term|short[- ]term|full[- ]time)\b',
        r'\b(?:user experience|customer satisfaction|market share|business model)\b',
        r'\b(?:supply chain|digital transformation|cloud computing|data analytics)\b',
        r'\b(?:research study|clinical trial|case study|pilot program)\b',
        r'\b(?:working memory|executive function|cognitive function|mental health)\b',
        r'\b(?:financial performance|revenue growth|profit margin|market cap)\b',
        r'\b\d+(?:\.\d+)?%\b',  # Percentages
        r'\$\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)?\b',  # Money
        r'\b\d+(?:,\d+)*\s*(?:participants|subjects|users|customers|employees)\b',  # Counts with units
    ]
    
    phrases = []
    for pattern in phrase_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phrases.extend([match.strip() for match in matches if len(match.strip()) > 3])
    
    return list(set(phrases))[:15]  # Limit to top 15 unique phrases

def extract_key_concepts(text: str, doc_type: str) -> dict:
    """Extract key concepts organized by category, preserving important phrases"""
    concepts = {
        'key_phrases': extract_important_phrases(text),
        'objectives': [],
        'methods': [],
        'processes': [],
        'benefits': [],
        'examples': [],
        'conclusions': [],
        'statistics': [],
        'outcomes': []
    }
    
    sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 15]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Prioritize sentences containing key phrases
        has_key_phrase = any(phrase.lower() in sentence_lower for phrase in concepts['key_phrases'])
        
        # Statistical information (high priority for preservation)
        if any(pattern in sentence for pattern in [' % ', ' percent', 'significantly', 'p<', 'p=', 'CI:', '95%']):
            concepts['statistics'].append(sentence)
        
        # Classify sentences by content type, preferring those with key phrases
        elif any(term in sentence_lower for term in ['objective', 'goal', 'aim', 'purpose']):
            concepts['objectives'].append(sentence)
        elif any(term in sentence_lower for term in ['method', 'technique', 'approach', 'way', 'process', 'strategy']):
            if has_key_phrase or len(sentence) > 50:
                concepts['methods'].append(sentence)
        elif any(term in sentence_lower for term in ['benefit', 'advantage', 'important', 'helps', 'improves', 'enhances']):
            concepts['benefits'].append(sentence)
        elif any(term in sentence_lower for term in ['result', 'outcome', 'effect', 'impact', 'finding', 'showed', 'demonstrated']):
            concepts['outcomes'].append(sentence)
        elif any(term in sentence_lower for term in ['conclusion', 'therefore', 'thus', 'overall', 'in summary']):
            concepts['conclusions'].append(sentence)
        elif any(term in sentence_lower for term in ['example', 'such as', 'like', 'including', 'for instance']):
            if has_key_phrase:
                concepts['examples'].append(sentence)
        elif len(sentence) > 60 and has_key_phrase:  # General important sentences with key phrases
            concepts['processes'].append(sentence)
    
    # Limit each category to avoid repetition, but keep statistics
    for key in concepts:
        if key == 'statistics':
            concepts[key] = concepts[key][:5]  # Keep more statistics
        elif key == 'key_phrases':
            continue  # Don't limit phrases
        else:
            concepts[key] = concepts[key][:4]
    
    return concepts

def improve_narrative_flow(text: str) -> str:
    """Improve flow between sentences and remove awkward transitions"""
    
    # Remove redundant sentence starters
    text = re.sub(r'\.\s+(Similarly|Additionally|Furthermore|Moreover),\s+', '. ', text)
    text = re.sub(r'\.\s+(This|These|It)\s+', '. ', text)
    text = re.sub(r'\s+(However|Nevertheless|Nonetheless),\s+', ', ', text)
    
    # Fix multiple spaces and clean punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    
    # Ensure proper sentence capitalization
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    cleaned_sentences = []
    
    for sentence in sentences:
        if sentence:
            # Capitalize first letter of each sentence
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            cleaned_sentences.append(sentence)
    
    # Rejoin with proper spacing
    result = '. '.join(cleaned_sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    return result

def create_bart_enhanced_summary(text: str, max_length: int) -> str:
    """Generate flowing narrative using BART with extractive enhancement"""
    
    try:
        # Split text into manageable chunks for BART
        words = text.split()
        chunk_size = 800  # Stay within BART's context window
        chunks = []
        
        # Create overlapping chunks to maintain context
        for i in range(0, len(words), chunk_size - 100):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        if not chunks:
            return None
            
        # Generate summaries for each chunk
        chunk_summaries = []
        target_length_per_chunk = max(max_length // len(chunks), 100)
        
        for i, chunk in enumerate(chunks):
            try:
                # Adjust parameters for better narrative flow
                result = summarizer(
                    chunk,
                    max_length=min(target_length_per_chunk + 50, 250),
                    min_length=min(target_length_per_chunk // 2, 50),
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    length_penalty=1.0  # Encourage complete sentences
                )
                
                if result and len(result) > 0 and 'summary_text' in result[0]:
                    summary_text = result[0]['summary_text'].strip()
                    
                    # Clean up BART artifacts
                    summary_text = re.sub(r'^(The|This|These)\s+', '', summary_text)
                    
                    if len(summary_text) > 30:  # Only add substantial summaries
                        chunk_summaries.append(summary_text)
                        logger.info(f"Generated BART chunk {i+1}: {len(summary_text)} chars")
                        
            except Exception as e:
                logger.warning(f"BART chunk {i+1} processing failed: {e}")
                continue
        
        if chunk_summaries:
            # Combine chunk summaries into flowing narrative
            if len(chunk_summaries) == 1:
                combined_summary = chunk_summaries[0]
            else:
                # Create transitions between chunks
                narrative_parts = []
                for i, summary in enumerate(chunk_summaries):
                    if i == 0:
                        narrative_parts.append(summary)
                    else:
                        # Add connecting phrases between chunks
                        connector = " Additionally, " if i == 1 else " Furthermore, "
                        
                        # Remove redundant sentence starts
                        clean_summary = re.sub(r'^(The|This|These|It)\s+', '', summary, flags=re.IGNORECASE)
                        clean_summary = clean_summary[0].lower() + clean_summary[1:] if clean_summary else summary
                        
                        narrative_parts.append(connector + clean_summary)
                
                combined_summary = "".join(narrative_parts)
            
            # Improve overall narrative flow
            combined_summary = improve_narrative_flow(combined_summary)
            
            # Trim to target length while preserving sentence boundaries
            if len(combined_summary) > max_length:
                sentences = combined_summary.split('. ')
                trimmed_sentences = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) + 2 <= max_length:
                        trimmed_sentences.append(sentence)
                        current_length += len(sentence) + 2
                    else:
                        break
                
                combined_summary = '. '.join(trimmed_sentences)
                if combined_summary and not combined_summary.endswith('.'):
                    combined_summary += '.'
            
            logger.info(f"BART enhanced summary generated: {len(combined_summary)} chars")
            return combined_summary
            
    except Exception as e:
        logger.error(f"BART enhanced summary failed: {e}")
        
    return None

def create_educational_narrative(concepts: dict, target_length: int) -> str:
    """Create flowing narrative for educational content"""
    
    paragraphs = []
    key_phrases = concepts.get('key_phrases', [])
    current_length = 0
    
    # Introduction with key phrases
    if key_phrases or concepts['objectives']:
        intro_phrases = [phrase for phrase in key_phrases[:3] if len(phrase) > 3]
        
        if intro_phrases:
            intro = f"This educational content covers {', '.join(intro_phrases)}. "
        else:
            intro = "This educational content provides essential learning concepts. "
            
        if concepts['objectives']:
            main_objective = concepts['objectives'][0]
            # Clean objective text
            main_objective = re.sub(r'(After reading this lesson,?\s*you will be able to:?\s*|OBJECTIVES\s*)', '', main_objective, flags=re.IGNORECASE).strip()
            if main_objective:
                intro += f"The primary focus is on {main_objective.lower()}. "
        
        paragraphs.append(intro)
        current_length += len(intro)
    
    # Methods and processes
    all_methods = concepts['methods'] + concepts['processes']
    if all_methods and current_length < target_length * 0.6:
        methods_content = []
        for method in all_methods[:4]:
            # Clean method descriptions
            cleaned = re.sub(r'^(For example,?|Similarly,?|Thus,?|Therefore,?)\s*', '', method, flags=re.IGNORECASE).strip()
            if len(cleaned) > 25:
                methods_content.append(cleaned)
        
        if methods_content:
            methods_para = f"Key methods include {methods_content[0].lower()}. "
            if len(methods_content) > 1:
                methods_para += f"Additionally, {'. '.join(methods_content[1:3])}. "
            
            paragraphs.append(methods_para)
            current_length += len(methods_para)
    
    # Benefits and outcomes
    all_benefits = concepts['benefits'] + concepts['outcomes']
    if all_benefits and current_length < target_length * 0.8:
        benefit_content = []
        for benefit in all_benefits[:3]:
            cleaned = re.sub(r'^(Thus,?|Therefore,?|In conclusion,?)\s*', '', benefit, flags=re.IGNORECASE).strip()
            if len(cleaned) > 20:
                benefit_content.append(cleaned)
        
        if benefit_content:
            benefits_para = f"Important benefits include {benefit_content[0].lower()}. "
            if len(benefit_content) > 1:
                benefits_para += f"Furthermore, {'. '.join(benefit_content[1:])}. "
            
            paragraphs.append(benefits_para)
            current_length += len(benefits_para)
    
    # Add conclusions and statistics
    remaining_content = concepts['conclusions'] + concepts['statistics']
    if remaining_content and current_length < target_length * 0.9:
        conclusion_text = []
        for item in remaining_content[:2]:
            cleaned = re.sub(r'^(In conclusion,?|Therefore,?|Thus,?)\s*', '', item, flags=re.IGNORECASE).strip()
            if len(cleaned) > 15:
                conclusion_text.append(cleaned)
        
        if conclusion_text:
            final_para = f"Key findings indicate that {conclusion_text[0].lower()}. "
            if len(conclusion_text) > 1:
                final_para += conclusion_text[1] + ". "
            paragraphs.append(final_para)
    
    # Combine and improve flow
    full_text = " ".join(paragraphs)
    return improve_narrative_flow(full_text)

def create_technical_narrative(concepts: dict, target_length: int) -> str:
    """Create flowing narrative for technical content"""
    
    paragraphs = []
    key_phrases = concepts.get('key_phrases', [])
    current_length = 0
    
    # Technical introduction
    tech_phrases = [phrase for phrase in key_phrases if any(term in phrase.lower() for term in ['system', 'method', 'approach', 'technology', 'process'])]
    
    if tech_phrases:
        intro = f"This technical content addresses {', '.join(tech_phrases[:2])}. "
    else:
        intro = "This technical analysis covers key methodological approaches. "
    
    if concepts['objectives']:
        intro += concepts['objectives'][0] + ". "
    
    paragraphs.append(intro)
    current_length += len(intro)
    
    # Methodology and processes
    method_content = concepts['methods'] + concepts['processes'][:2]
    if method_content and current_length < target_length * 0.6:
        methods_text = []
        for method in method_content[:3]:
            cleaned = re.sub(r'^(The\s+|This\s+)', '', method).strip()
            if len(cleaned) > 30:
                methods_text.append(cleaned)
        
        if methods_text:
            method_para = f"The implementation involves {methods_text[0].lower()}. "
            if len(methods_text) > 1:
                method_para += f"Additional processes include {'. '.join(methods_text[1:])}. "
            paragraphs.append(method_para)
            current_length += len(method_para)
    
    # Outcomes and statistics
    results = concepts['outcomes'] + concepts['statistics']
    if results and current_length < target_length * 0.8:
        result_text = []
        for result in results[:3]:
            if len(result.strip()) > 20:
                result_text.append(result.strip())
        
        if result_text:
            results_para = f"Results demonstrate that {result_text[0].lower()}. "
            if len(result_text) > 1:
                results_para += f"Additionally, {'. '.join(result_text[1:])}. "
            paragraphs.append(results_para)
    
    full_text = " ".join(paragraphs)
    return improve_narrative_flow(full_text)

def create_business_narrative(concepts: dict, target_length: int) -> str:
    """Create flowing narrative for business content"""
    
    paragraphs = []
    key_phrases = concepts.get('key_phrases', [])
    current_length = 0
    
    # Business introduction with financial/performance focus
    business_phrases = [phrase for phrase in key_phrases if any(term in phrase.lower() for term in ['growth', 'revenue', 'performance', 'market', 'business'])]
    
    if business_phrases:
        intro = f"This business analysis examines {', '.join(business_phrases[:2])}. "
    else:
        intro = "This business report analyzes key performance indicators. "
    
    paragraphs.append(intro)
    current_length += len(intro)
    
    # Key findings and statistics (priority for business)
    findings = concepts['statistics'] + concepts['outcomes']
    if findings and current_length < target_length * 0.6:
        finding_text = []
        for finding in findings[:4]:
            if any(char.isdigit() for char in finding) or len(finding) > 40:  # Prioritize data-rich content
                finding_text.append(finding.strip())
        
        if finding_text:
            findings_para = f"Key performance metrics show {finding_text[0].lower()}. "
            if len(finding_text) > 1:
                findings_para += f"Furthermore, {'. '.join(finding_text[1:3])}. "
            paragraphs.append(findings_para)
            current_length += len(findings_para)
    
    # Strategic methods and benefits
    strategy_content = concepts['methods'] + concepts['benefits']
    if strategy_content and current_length < target_length * 0.8:
        strategy_text = []
        for item in strategy_content[:3]:
            cleaned = re.sub(r'^(The company|The organization)\s+', '', item, flags=re.IGNORECASE).strip()
            if len(cleaned) > 25:
                strategy_text.append(cleaned)
        
        if strategy_text:
            strategy_para = f"Strategic initiatives include {strategy_text[0].lower()}. "
            if len(strategy_text) > 1:
                strategy_para += f"These efforts also {'. '.join(strategy_text[1:])}. "
            paragraphs.append(strategy_para)
    
    full_text = " ".join(paragraphs)
    return improve_narrative_flow(full_text)

def create_news_narrative(concepts: dict, target_length: int) -> str:
    """Create flowing narrative for news content"""
    
    paragraphs = []
    key_phrases = concepts.get('key_phrases', [])
    current_length = 0
    
    # News introduction
    if key_phrases:
        intro = f"Recent developments regarding {', '.join(key_phrases[:2])} have emerged. "
    else:
        intro = "Recent news developments have been reported. "
    
    paragraphs.append(intro)
    current_length += len(intro)
    
    # Main findings and outcomes (what happened)
    main_content = concepts['outcomes'] + concepts['statistics'] + concepts['methods']
    if main_content and current_length < target_length * 0.7:
        news_text = []
        for item in main_content[:4]:
            # Clean news-specific artifacts
            cleaned = re.sub(r'^(According to|Officials said|Reports indicate)\s*', '', item, flags=re.IGNORECASE).strip()
            if len(cleaned) > 25:
                news_text.append(cleaned)
        
        if news_text:
            news_para = f"The situation involves {news_text[0].lower()}. "
            if len(news_text) > 1:
                news_para += f"Additionally, {'. '.join(news_text[1:3])}. "
            paragraphs.append(news_para)
            current_length += len(news_para)
    
    # Impact and conclusions
    impact_content = concepts['benefits'] + concepts['conclusions']
    if impact_content and current_length < target_length * 0.9:
        impact_text = []
        for item in impact_content[:2]:
            if len(item.strip()) > 20:
                impact_text.append(item.strip())
        
        if impact_text:
            impact_para = f"The implications suggest {impact_text[0].lower()}. "
            if len(impact_text) > 1:
                impact_para += impact_text[1] + ". "
            paragraphs.append(impact_para)
    
    full_text = " ".join(paragraphs)
    return improve_narrative_flow(full_text)

def create_general_narrative(concepts: dict, target_length: int) -> str:
    """Create flowing narrative for general content"""
    
    paragraphs = []
    key_phrases = concepts.get('key_phrases', [])
    current_length = 0
    
    # General introduction
    if key_phrases:
        intro = f"This content examines {', '.join(key_phrases[:2])}. "
    else:
        intro = "This content covers several important topics. "
    
    paragraphs.append(intro)
    current_length += len(intro)
    
    # Combine all content types for general narrative
    all_content = []
    for category in ['methods', 'processes', 'benefits', 'outcomes', 'statistics']:
        all_content.extend(concepts[category])
    
    if all_content and current_length < target_length * 0.7:
        # Create 2 main content paragraphs
        mid_point = len(all_content) // 2
        first_half = all_content[:mid_point]
        second_half = all_content[mid_point:]
        
        # First content paragraph
        if first_half:
            first_content = []
            for item in first_half[:3]:
                cleaned = re.sub(r'^(The|This|These)\s+', '', item).strip()
                if len(cleaned) > 20:
                    first_content.append(cleaned)
            
            if first_content:
                first_para = f"Key aspects include {first_content[0].lower()}. "
                if len(first_content) > 1:
                    first_para += f"Moreover, {'. '.join(first_content[1:])}. "
                paragraphs.append(first_para)
                current_length += len(first_para)
        
        # Second content paragraph
        if second_half and current_length < target_length * 0.9:
            second_content = []
            for item in second_half[:3]:
                cleaned = re.sub(r'^(Additionally|Furthermore|Moreover)\s*,?\s*', '', item, flags=re.IGNORECASE).strip()
                if len(cleaned) > 20:
                    second_content.append(cleaned)
            
            if second_content:
                second_para = f"Important considerations include {second_content[0].lower()}. "
                if len(second_content) > 1:
                    second_para += f"These factors {'. '.join(second_content[1:])}. "
                paragraphs.append(second_para)
    
    # Add conclusions if space permits
    if concepts['conclusions'] and current_length < target_length * 0.95:
        conclusion = concepts['conclusions'][0]
        cleaned_conclusion = re.sub(r'^(In conclusion,?|Therefore,?|Thus,?)\s*', '', conclusion, flags=re.IGNORECASE).strip()
        if cleaned_conclusion:
            conclusion_para = f"Overall, {cleaned_conclusion.lower()}. "
            paragraphs.append(conclusion_para)
    
    full_text = " ".join(paragraphs)
    return improve_narrative_flow(full_text)

def create_coherent_summary(text: str, doc_type: str, target_length: int) -> str:
    """Create a coherent, flowing summary using extractive methods"""
    
    # Extract key concepts with phrase preservation
    key_concepts = extract_key_concepts(text, doc_type)
    
    # Build narrative based on document type
    if doc_type == "educational":
        return create_educational_narrative(key_concepts, target_length)
    elif doc_type == "technical":
        return create_technical_narrative(key_concepts, target_length)
    elif doc_type == "business":
        return create_business_narrative(key_concepts, target_length)
    elif doc_type == "news":
        return create_news_narrative(key_concepts, target_length)
    else:
        return create_general_narrative(key_concepts, target_length)

def create_hybrid_summary(text: str, doc_type: str, max_length: int, min_length: int) -> str:
    """Create hybrid summary: prioritize coherent narrative with BART enhancement"""
    
    try:
        # STEP 1: Try BART first if available and text is substantial
        if summarizer and len(text) > 300:
            bart_summary = create_bart_enhanced_summary(text, max_length)
            if bart_summary and len(bart_summary) >= min_length:
                logger.info("Used BART-generated narrative summary")
                return bart_summary
    
        # STEP 2: Fallback to coherent extractive narrative
        logger.info("Using coherent extractive narrative")
        return create_coherent_summary(text, doc_type, max_length)
        
    except Exception as e:
        logger.warning(f"Hybrid summary failed: {e}")
        return create_coherent_summary(text, doc_type, max_length)

def generate_summary(text: str, max_length: int = 150, min_length: int = 30) -> dict:
    """Generate hybrid summary using both AI (generative) and extractive methods"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")
    
    # Clean and prepare text
    text = text.strip()
    word_count = len(text.split())
    
    if word_count < 10:
        return {
            "summary": text,
            "original_length": len(text),
            "summary_length": len(text),
            "compression_ratio": 1.0
        }
    
    try:
        # Detect document type for adaptive processing
        doc_type = detect_document_type(text)
        logger.info(f"Detected document type: {doc_type}")
        
        # Use hybrid approach combining both methods
        summary = create_hybrid_summary(text, doc_type, max_length, min_length)
        
        # Validate and clean summary
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, summary) and not re.search(email_pattern, text):
            logger.warning("Summary contains email not in original text, using extractive fallback")
            summary = create_coherent_summary(text, doc_type, max_length)
        
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(text), 3)
        }
        
    except Exception as e:
        logger.error(f"Failed to generate hybrid summary: {e}")
        # Final fallback to extractive only
        doc_type = detect_document_type(text)
        summary = create_coherent_summary(text, doc_type, max_length)
        logger.info("Used coherent extractive summary as fallback")
        
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(text), 3)
        }

# Web content extraction functions
def detect_web_content_type(url: str, soup: BeautifulSoup) -> str:
    """Detect the type of web content"""
    url_lower = url.lower()
    
    # Check for academic/research content
    if any(domain in url_lower for domain in ['ncbi.nlm.nih.gov', 'pubmed', 'arxiv.org', 'sciencedirect', 'springer']):
        return "research_article"
    
    # Check for news sites
    if any(domain in url_lower for domain in ['cnn.com', 'bbc.com', 'reuters.com', 'news', 'times']):
        return "news"
    
    # Check meta tags and content for more clues
    meta_tags = soup.find_all('meta')
    for meta in meta_tags:
        name = meta.get('name', '').lower()
        content = meta.get('content', '').lower()
        
        if name in ['article:section', 'article:tag'] or 'research' in content or 'study' in content:
            return "research_article"
        elif 'news' in content or 'breaking' in content:
            return "news"
    
    # Check for academic indicators in content
    text_content = soup.get_text().lower()
    if any(term in text_content for term in ['abstract', 'methodology', 'results', 'conclusion', 'references', 'doi:']):
        return "research_article"
    
    # Check for blog indicators
    if any(term in url_lower for term in ['blog', 'medium.com', 'substack']):
        return "blog"
    
    # Check for business content
    if any(term in text_content for term in ['company', 'business', 'market', 'revenue', 'financial']):
        return "business"
    
    return "general"

def extract_main_content(soup: BeautifulSoup, content_type: str) -> str:
    """Extract main content based on content type"""
    
    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
        element.decompose()
    
    content = ""
    
    if content_type == "research_article":
        content = extract_research_article_content(soup)
    elif content_type == "news":
        content = extract_news_content(soup)
    elif content_type == "blog":
        content = extract_blog_content(soup)
    else:
        content = extract_general_content(soup)
    
    return content

def extract_research_article_content(soup: BeautifulSoup) -> str:
    """Extract content from research articles"""
    content_parts = []
    
    # Try to find abstract
    abstract_selectors = [
        'div[class*="abstract"]', 'section[class*="abstract"]', 
        'div#abstract', 'section#abstract',
        'h2:contains("Abstract") + *', 'h3:contains("Abstract") + *'
    ]
    
    for selector in abstract_selectors:
        try:
            abstract = soup.select(selector)
            if abstract:
                content_parts.append("ABSTRACT:")
                content_parts.append(abstract[0].get_text(strip=True))
                break
        except:
            continue
    
    # Try to find main article content
    article_selectors = [
        'div[class*="article-body"]', 'div[class*="content"]', 
        'article', 'main', 'div[class*="text"]',
        'div.article-text', 'div.main-content'
    ]
    
    for selector in article_selectors:
        try:
            article_content = soup.select(selector)
            if article_content and len(article_content[0].get_text(strip=True)) > 200:
                content_parts.append("MAIN CONTENT:")
                content_parts.append(article_content[0].get_text(separator=' ', strip=True))
                break
        except:
            continue
    
    # If no specific selectors worked, get paragraphs
    if len(content_parts) < 2:
        paragraphs = soup.find_all('p')
        if paragraphs:
            content_parts.append(' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]))
    
    return '\n\n'.join(content_parts)

def extract_news_content(soup: BeautifulSoup) -> str:
    """Extract content from news articles"""
    content_parts = []
    
    # Try news-specific selectors
    news_selectors = [
        'div[class*="story-body"]', 'div[class*="article-body"]',
        'div[class*="content"]', 'article', 'main'
    ]
    
    for selector in news_selectors:
        try:
            news_content = soup.select(selector)
            if news_content and len(news_content[0].get_text(strip=True)) > 100:
                content_parts.append(news_content[0].get_text(separator=' ', strip=True))
                break
        except:
            continue
    
    # Fallback to paragraphs
    if not content_parts:
        paragraphs = soup.find_all('p')
        if paragraphs:
            content_parts.append(' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]))
    
    return '\n\n'.join(content_parts)

def extract_blog_content(soup: BeautifulSoup) -> str:
    """Extract content from blog posts"""
    content_parts = []
    
    # Try blog-specific selectors
    blog_selectors = [
        'div[class*="post-content"]', 'div[class*="entry-content"]',
        'div[class*="content"]', 'article', 'main'
    ]
    
    for selector in blog_selectors:
        try:
            blog_content = soup.select(selector)
            if blog_content and len(blog_content[0].get_text(strip=True)) > 100:
                content_parts.append(blog_content[0].get_text(separator=' ', strip=True))
                break
        except:
            continue
    
    # Fallback to paragraphs
    if not content_parts:
        paragraphs = soup.find_all('p')
        if paragraphs:
            content_parts.append(' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]))
    
    return '\n\n'.join(content_parts)

def extract_general_content(soup: BeautifulSoup) -> str:
    """Extract general content from any webpage"""
    content_parts = []
    
    # Try common content selectors
    general_selectors = [
        'main', 'article', 'div[class*="content"]', 
        'div[class*="body"]', 'div[role="main"]'
    ]
    
    for selector in general_selectors:
        try:
            general_content = soup.select(selector)
            if general_content and len(general_content[0].get_text(strip=True)) > 100:
                content_parts.append(general_content[0].get_text(separator=' ', strip=True))
                break
        except:
            continue
    
    # Fallback: get all paragraphs and headings
    if not content_parts:
        elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if elements:
            content_parts.append(' '.join([elem.get_text(strip=True) for elem in elements if len(elem.get_text(strip=True)) > 20]))
    
    return '\n\n'.join(content_parts)

def clean_extracted_content(content: str) -> str:
    """Clean extracted web content"""
    if not content:
        return ""
    
    # Remove extra whitespace and normalize
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    # Remove common web artifacts
    content = re.sub(r'Cookie|Accept|Privacy Policy|Terms of Service', '', content, flags=re.IGNORECASE)
    content = re.sub(r'Click here|Read more|Continue reading', '', content, flags=re.IGNORECASE)
    
    # Remove URLs and email addresses that might be artifacts
    content = re.sub(r'http[s]?://\S+', '', content)
    content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', content)
    
    # Clean up multiple spaces again
    content = re.sub(r'\s+', ' ', content)
    
    return content.strip()

def fetch_web_content(url: str) -> dict:
    """Fetch and extract content from web URL"""
    try:
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        logger.info(f"Fetching content from: {url}")
        
        # Make the request with timeout
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'text/html' not in content_type and 'application/xml' not in content_type:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported content type: {content_type}. Only HTML and XML content are supported."
            )
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No title found"
        
        # Detect content type based on URL and content
        detected_type = detect_web_content_type(url, soup)
        
        # Extract main content using various strategies
        content = extract_main_content(soup, detected_type)
        
        if not content or len(content.strip()) < 50:
            # Fallback: extract all text from body
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)
            else:
                content = soup.get_text(separator=' ', strip=True)
        
        # Clean the content
        content = clean_extracted_content(content)
        
        if not content or len(content.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract sufficient readable content from the webpage"
            )
        
        logger.info(f"Successfully extracted {len(content)} characters from {url}")
        
        return {
            "content": content,
            "title": title,
            "url": url,
            "content_type": detected_type,
            "original_url": url
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timeout - the webpage took too long to respond")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to the webpage")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Webpage not found")
        elif e.response.status_code == 403:
            raise HTTPException(status_code=403, detail="Access denied to the webpage")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Failed to fetch content from {url}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch or process the webpage content")

# PDF processing functions
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text += page.extract_text() + "\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue
        
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to process PDF file")

def extract_text_from_pdf_by_pages(pdf_file) -> dict:
    """Extract text from PDF file page by page"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_text = {}
        total_pages = len(pdf_reader.pages)
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only include pages with content
                    pages_text[page_num + 1] = page_text.strip()
                    logger.info(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        return {
            "pages": pages_text,
            "total_pages": total_pages,
            "content_pages": len(pages_text)
        }
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to process PDF file")

def create_page_wise_summary(pages_data: dict, doc_type: str, target_total_length: int) -> dict:
    """Create page-wise summaries and overall summary using enhanced methods"""
    
    pages_text = pages_data["pages"]
    total_content_length = sum(len(text) for text in pages_text.values())
    
    # Calculate target length per page based on content proportion
    page_summaries = {}
    
    for page_num, page_text in pages_text.items():
        # Calculate target length for this page (proportional to content)
        page_proportion = len(page_text) / total_content_length
        page_target_length = int(target_total_length * page_proportion)
        page_target_length = max(100, page_target_length)  # Minimum 100 chars per page
        
        try:
            # Create summary for this page using enhanced method
            raw_summary = create_hybrid_summary(page_text, doc_type, page_target_length, 50)
            
            # Add page number prefix to the summary
            page_summary = f"Page {page_num}: {raw_summary}"
            
            page_summaries[page_num] = {
                "original_length": len(page_text),
                "summary": page_summary,
                "summary_length": len(page_summary),
                "compression_ratio": round(len(page_summary) / len(page_text), 3)
            }
            logger.info(f"Page {page_num}: {len(page_text)} -> {len(page_summary)} chars")
        except Exception as e:
            logger.warning(f"Failed to summarize page {page_num}: {e}")
            # Fallback to simple truncation with page number
            fallback_content = page_text[:page_target_length] + "..." if len(page_text) > page_target_length else page_text
            page_summary = f"Page {page_num}: {fallback_content}"
            page_summaries[page_num] = {
                "original_length": len(page_text),
                "summary": page_summary,
                "summary_length": len(page_summary),
                "compression_ratio": round(len(page_summary) / len(page_text), 3)
            }
    
    # Create overall summary from page summaries with clear page separation
    page_summary_parts = []
    for page_num in sorted(page_summaries.keys()):
        page_summary_parts.append(page_summaries[page_num]["summary"])
    
    all_page_summaries = "\n\n".join(page_summary_parts)
    
    # If overall summary is too long, create a meta-summary but preserve page structure
    if len(all_page_summaries) > target_total_length:
        # Create condensed version while keeping page numbers
        condensed_parts = []
        for page_num in sorted(page_summaries.keys()):
            original_summary = page_summaries[page_num]["summary"]
            # Extract content after "Page X: " prefix
            content = original_summary.split(": ", 1)[1] if ": " in original_summary else original_summary
            # Create shorter version
            condensed_content = create_hybrid_summary(content, doc_type, target_total_length // len(page_summaries), 30)
            condensed_parts.append(f"Page {page_num}: {condensed_content}")
        
        overall_summary = "\n\n".join(condensed_parts)
    else:
        overall_summary = all_page_summaries
    
    return {
        "page_summaries": page_summaries,
        "overall_summary": overall_summary,
        "total_pages": pages_data["total_pages"],
        "content_pages": pages_data["content_pages"],
        "total_original_length": total_content_length,
        "total_summary_length": len(overall_summary),
        "overall_compression_ratio": round(len(overall_summary) / total_content_length, 3)
    }

# Utility functions
def clean_text_for_json(text: str) -> str:
    """Clean text to make it JSON-safe while preserving content"""
    if not text:
        return ""
    
    # Remove control characters that break JSON parsing
    # Keep \n (newline) and \t (tab) but remove other control chars
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)
    
    # Replace problematic characters that can break JSON
    cleaned = cleaned.replace('\r\n', '\n')  # Normalize line endings
    cleaned = cleaned.replace('\r', '\n')    # Convert old Mac line endings
    
    # Clean up multiple spaces but preserve single newlines
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces/tabs to single space
    cleaned = re.sub(r'\n+', '\n', cleaned)   # Multiple newlines to single newline
    
    return cleaned.strip()

# Middleware and API endpoints
@app.middleware("http")
async def process_text_requests(request: Request, call_next):
    """Middleware to handle problematic characters in text requests"""
    if request.url.path == "/summarize/text" and request.method == "POST":
        # Get the raw body
        body = await request.body()
        
        try:
            # Try to decode and clean the body
            body_str = body.decode('utf-8')
            cleaned_body = clean_text_for_json(body_str)
            
            # Try to parse JSON to validate
            try:
                data = json.loads(cleaned_body)
                # If successful, create a new request with cleaned body
                new_body = json.dumps(data).encode('utf-8')
                request._body = new_body
                
                # Update content length
                request.headers.__dict__["_list"] = [
                    (name, value) for name, value in request.headers.items() 
                    if name.lower() != "content-length"
                ]
                request.headers.__dict__["_list"].append(
                    ("content-length", str(len(new_body)))
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, try manual extraction
                text_match = re.search(r'"text"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', cleaned_body, re.DOTALL)
                if text_match:
                    extracted_text = text_match.group(1)
                    extracted_text = extracted_text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                    data = {"text": extracted_text}
                    new_body = json.dumps(data).encode('utf-8')
                    request._body = new_body
        except Exception:
            # If all fails, let the original request through
            pass
    
    response = await call_next(request)
    return response

@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {"message": "Text & PDF Summarizer API v2.0 is running!", "status": "healthy"}

@app.post("/summarize/text", response_model=SummaryResponse, tags=["Text Summarization"])
async def summarize_text(request: TextSummaryRequest):
    """
    Summarize Text Content - Enhanced v2.0
    
    **Improved summarization with better ROUGE scores and narrative flow**
    **Combines advanced AI (BART) with intelligent extractive methods**
    **Automatic length calculation - summary will be approximately 40% of original text length**
    
    **Key Improvements in v2.0:**
    - Enhanced phrase preservation for better ROUGE-2 scores
    - Improved narrative flow for better ROUGE-L scores  
    - Advanced BART integration with chunk processing
    - Content-type specific narrative construction
    - Better handling of technical terminology and statistics
    
    **Features:**
    - Works with Swagger UI interface (paste text directly)
    - Handles text with newlines, quotes, special characters
    - Preserves important multi-word phrases and statistics
    - Adapts to different document types (educational, technical, business, etc.)
    - Generates flowing narratives instead of bullet points
    
    **How it works:**
    1. Document Type Detection: Identifies content type for optimal processing
    2. Key Phrase Extraction: Preserves important multi-word terms
    3. BART Enhancement: Uses AI for natural language generation
    4. Narrative Construction: Creates flowing, coherent summaries
    5. Flow Optimization: Improves sentence transitions and readability
    
    **Supported text types:**
    - Educational content (lessons, tutorials, guides)
    - Technical documents (research, analysis, methodologies) 
    - Business reports (financials, strategies, performance)
    - News articles (breaking news, events, announcements)
    - Research articles (studies, experiments, findings)
    
    **Request Body:**
    ```json
    {
        "text": "Your text content to summarize..."
    }
    ```
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        # Clean the text to handle any problematic characters
        cleaned_text = clean_text_for_json(request.text)
        
        if not cleaned_text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty after cleaning")
        
        logger.info(f"Processing text input of {len(cleaned_text)} characters")
        
        # Calculate summary length as 40% of original text
        original_length = len(cleaned_text)
        target_summary_length = int(original_length * 0.4)
        
        # Use the target length directly with minimal bounds
        max_length = max(target_summary_length, 200)  # Minimum 200 chars for readability
        min_length = max(int(target_summary_length * 0.7), 100)  # 70% of target, min 100
        
        result = generate_summary(
            text=cleaned_text,
            max_length=max_length,
            min_length=min_length
        )
        
        logger.info(f"Enhanced hybrid summary generated successfully. Target: {target_summary_length}, Actual: {result['summary_length']}")
        return SummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process text summarization: {str(e)}")

@app.post("/summarize/url", response_model=URLSummaryResponse, tags=["URL Summarization"])
async def summarize_url(request: URLSummaryRequest):
    """
    Summarize Content from Web URL - Enhanced v2.0
    
    **Fetch and summarize content from any public web URL with improved quality**
    **Enhanced content extraction and intelligent summarization with better ROUGE scores**
    
    **Key Improvements in v2.0:**
    - Better content extraction with phrase preservation
    - Enhanced narrative flow for web content
    - Improved handling of different website types
    - Advanced document type detection for web content
    
    **Features:**
    - Fetches content from any public HTTP/HTTPS URL
    - Intelligent content extraction based on page type
    - Supports academic articles (PubMed, arXiv, etc.)
    - Handles news articles, blogs, and general web content
    - Automatic document type detection for optimal summarization
    - 40% compression ratio with improved readability
    
    **Supported URL types:**
    - Research articles (PubMed, NCBI, arXiv, ScienceDirect)
    - News websites (CNN, BBC, Reuters, etc.)
    - Blog posts (Medium, Substack, personal blogs)
    - Business websites and reports
    - Educational content and documentation
    - General web pages with text content
    
    **Request Body:**
    ```json
    {
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4063875/"
    }
    ```
    """
    try:
        if not request.url or not request.url.strip():
            raise HTTPException(status_code=400, detail="URL cannot be empty")
        
        url = request.url.strip()
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logger.info(f"Processing URL: {url}")
        
        # Fetch web content
        web_data = fetch_web_content(url)
        
        content = web_data["content"]
        title = web_data["title"]
        content_type = web_data["content_type"]
        
        logger.info(f"Extracted {len(content)} characters from {url}")
        logger.info(f"Detected content type: {content_type}")
        logger.info(f"Page title: {title}")
        
        # Calculate summary length as 40% of original content
        original_length = len(content)
        target_summary_length = int(original_length * 0.4)
        
        # Set bounds for web content (typically longer than manual text input)
        max_length = max(target_summary_length, 300)  # Minimum 300 chars for web content
        min_length = max(int(target_summary_length * 0.7), 150)  # 70% of target, min 150
        
        # Generate summary using the enhanced hybrid approach
        summary_result = generate_summary(
            text=content,
            max_length=max_length,
            min_length=min_length
        )
        
        logger.info(f"URL summary generated successfully. Target: {target_summary_length}, Actual: {summary_result['summary_length']}")
        
        return URLSummaryResponse(
            summary=summary_result["summary"],
            original_length=summary_result["original_length"],
            summary_length=summary_result["summary_length"],
            compression_ratio=summary_result["compression_ratio"],
            url=url,
            title=title,
            content_type=content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in URL summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL summarization: {str(e)}")

@app.post("/summarize/pdf-pages", response_model=PageWiseSummaryResponse, tags=["PDF Summarization"])
async def summarize_pdf_pages(
    file: UploadFile = File(..., description="PDF file to summarize page by page")
):
    """
    Summarize PDF File with Page-wise Breakdown - Enhanced v2.0
    
    **Enhanced page-by-page summaries with improved narrative flow and ROUGE scores**
    **Each page summary uses advanced summarization with phrase preservation**
    
    **Key Improvements in v2.0:**
    - Enhanced narrative construction for each page
    - Better phrase preservation across pages
    - Improved overall document coherence
    - Advanced content-type detection for PDFs
    
    **Features:**
    - Page-by-page content extraction and enhanced summarization
    - Proportional summary lengths based on page content
    - Overall document summary combining all pages
    - Detailed statistics for each page and overall document
    - Handles multi-page documents with varying content density
    - Uses BART and extractive methods for optimal quality
    
    **Response includes:**
    - Individual enhanced summary for each page
    - Coherent overall document summary
    - Page statistics (original length, summary length, compression ratio)
    - Document metadata (total pages, content pages)
    
    **Example Usage:**
    ```bash
    curl -X POST "http://localhost:8000/summarize/pdf-pages" \
         -F "file=@document.pdf"
    ```
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported. Please upload a .pdf file."
            )
        
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail="File size too large (maximum 10MB allowed)"
            )
        
        logger.info(f"Processing PDF file page-wise: {file.filename} ({file.size} bytes)")
        
        # Read and process PDF page by page
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        pages_data = extract_text_from_pdf_by_pages(pdf_file)
        
        if not pages_data["pages"]:
            raise HTTPException(
                status_code=400, 
                detail="No readable text found in the PDF file."
            )
        
        logger.info(f"Extracted text from {pages_data['content_pages']} out of {pages_data['total_pages']} pages")
        
        # Calculate target total length (40% of all content)
        total_length = sum(len(text) for text in pages_data["pages"].values())
        target_total_length = int(total_length * 0.4)
        
        # Create page-wise summaries using enhanced methods
        result = create_page_wise_summary(pages_data, "educational", target_total_length)
        
        logger.info(f"Generated enhanced page-wise summaries. Total: {result['total_original_length']} -> {result['total_summary_length']} chars")
        
        return PageWiseSummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in page-wise PDF summarization: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to process PDF summarization"
        )

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Detailed health check with model status and v2.0 features"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "model_loaded": summarizer is not None,
        "model_info": {
            "name": "facebook/bart-large-cnn",
            "type": "Transformer-based summarization",
            "enhanced_features": [
                "Chunk processing for long texts",
                "Narrative flow optimization", 
                "Multi-word phrase preservation",
                "Content-type adaptive processing"
            ]
        },
        "endpoints": {
            "text_summarization": "/summarize/text",
            "url_summarization": "/summarize/url",
            "pdf_summarization": "/summarize/pdf-pages"
        },
        "improvements_v2": {
            "rouge_optimization": "Enhanced ROUGE-1, ROUGE-2, and ROUGE-L scores",
            "narrative_flow": "Improved sentence transitions and coherence",
            "phrase_preservation": "Better handling of technical terms and statistics", 
            "bart_integration": "Advanced AI processing with chunk handling",
            "content_adaptation": "Document-type specific narrative construction"
        },
        "features": {
            "text_input": "JSON payload with text content",
            "url_input": "JSON payload with web URL",
            "pdf_input": "Multipart form with PDF file upload",
            "max_file_size": "10MB",
            "supported_formats": [".pdf"],
            "supported_urls": ["HTTP", "HTTPS", "Research articles", "News", "Blogs"],
            "compression_ratio": "~40% of original length",
            "output_format": "Flowing narrative (not bullet points)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )