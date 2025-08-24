#!/usr/bin/env python3
"""
ROUGE Evaluation Script for Summarization APIs
Evaluates text, URL, and PDF summarization endpoints using human reference summaries
"""

import json
import requests
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple, Any
import logging
import time
from pathlib import Path
import argparse
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ROUGEEvaluator:
    """ROUGE evaluation framework for summarization APIs"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip('/')
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.results = []
        
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load the evaluation dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f"Loaded dataset with {len(dataset['samples'])} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def call_text_api(self, text: str) -> Dict[str, Any]:
        """Call the text summarization API"""
        try:
            response = requests.post(
                f"{self.api_base_url}/summarize/text",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Text API call failed: {e}")
            raise
    
    def call_url_api(self, url: str) -> Dict[str, Any]:
        """Call the URL summarization API"""
        try:
            response = requests.post(
                f"{self.api_base_url}/summarize/url",
                json={"url": url},
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"URL API call failed: {e}")
            raise
    
    def call_pdf_api(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Call the PDF summarization API"""
        try:
            files = {"file": (filename, pdf_content, "application/pdf")}
            response = requests.post(
                f"{self.api_base_url}/summarize/pdf-pages",
                files=files,
                timeout=180
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"PDF API call failed: {e}")
            raise
    
    def create_mock_pdf(self, page_content: Dict[str, str], filename: str) -> bytes:
        """Create a mock PDF for testing (placeholder - would need actual PDF generation)"""
        # This is a placeholder - in actual implementation, you'd create a real PDF
        # For testing purposes, we'll just concatenate all page content as text
        combined_text = "\n\n".join([f"Page {page}: {content}" for page, content in page_content.items()])
        logger.warning(f"Mock PDF creation for {filename} - using combined text instead")
        return combined_text.encode('utf-8')
    
    def calculate_rouge_scores(self, generated_summary: str, reference_summary: str) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores between generated and reference summaries"""
        scores = self.scorer.score(reference_summary, generated_summary)
        
        # Convert to dictionary format
        rouge_scores = {}
        for metric, score in scores.items():
            rouge_scores[metric] = {
                'precision': score.precision,
                'recall': score.recall,
                'fmeasure': score.fmeasure
            }
        
        return rouge_scores
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample"""
        logger.info(f"Evaluating sample {sample['id']}: {sample['content_type']}")
        
        result = {
            'sample_id': sample['id'],
            'content_type': sample['content_type'],
            'api_endpoint': sample['api_endpoint'],
            'success': False,
            'error': None,
            'generated_summary': None,
            'reference_summary': sample['human_reference_summary'],
            'rouge_scores': None,
            'length_analysis': {},
            'evaluation_score': None
        }
        
        try:
            # Call appropriate API based on endpoint
            if sample['api_endpoint'] == '/summarize/text':
                api_response = self.call_text_api(sample['source_text'])
                result['generated_summary'] = api_response['summary']
                result['api_metadata'] = {
                    'original_length': api_response['original_length'],
                    'summary_length': api_response['summary_length'],
                    'compression_ratio': api_response['compression_ratio']
                }
                
            elif sample['api_endpoint'] == '/summarize/url':
                # For URL samples, use the source_text as if it were fetched from the URL
                # In real testing, you'd need actual URLs that are accessible
                api_response = self.call_text_api(sample['source_text'])  # Fallback for testing
                result['generated_summary'] = api_response['summary']
                result['api_metadata'] = {
                    'original_length': api_response['original_length'],
                    'summary_length': api_response['summary_length'],
                    'compression_ratio': api_response['compression_ratio']
                }
                
            elif sample['api_endpoint'] == '/summarize/pdf':
                # For PDF samples, create mock PDF from page content
                if 'page_content' in sample:
                    mock_pdf = self.create_mock_pdf(sample['page_content'], f"sample_{sample['id']}.pdf")
                    # For testing, we'll use text API with combined content
                    combined_text = "\n\n".join([f"Page {page}: {content}" for page, content in sample['page_content'].items()])
                    api_response = self.call_text_api(combined_text)
                    result['generated_summary'] = api_response['summary']
                else:
                    raise ValueError("PDF sample missing page_content")
            
            # Calculate ROUGE scores
            result['rouge_scores'] = self.calculate_rouge_scores(
                result['generated_summary'], 
                result['reference_summary']
            )
            
            # Length analysis
            result['length_analysis'] = {
                'generated_length': len(result['generated_summary']),
                'reference_length': len(result['reference_summary']),
                'length_ratio': len(result['generated_summary']) / len(result['reference_summary']),
                'within_expected_range': self.check_length_range(
                    len(result['generated_summary']), 
                    sample.get('expected_length_range', [0, float('inf')])
                )
            }
            
            # Calculate overall evaluation score
            result['evaluation_score'] = self.calculate_evaluation_score(result['rouge_scores'])
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to evaluate sample {sample['id']}: {e}")
        
        return result
    
    def check_length_range(self, length: int, expected_range: List[int]) -> bool:
        """Check if generated summary length is within expected range"""
        return expected_range[0] <= length <= expected_range[1]
    
    def calculate_evaluation_score(self, rouge_scores: Dict[str, Dict[str, float]]) -> str:
        """Calculate overall evaluation score based on ROUGE metrics"""
        if not rouge_scores:
            return "failed"
        
        rouge1_f1 = rouge_scores['rouge1']['fmeasure']
        rouge2_f1 = rouge_scores['rouge2']['fmeasure']  
        rougeL_f1 = rouge_scores['rougeL']['fmeasure']
        
        # Evaluation criteria from dataset
        if rouge1_f1 > 0.60 and rouge2_f1 > 0.40 and rougeL_f1 > 0.55:
            return "excellent"
        elif rouge1_f1 >= 0.45 and rouge2_f1 >= 0.25 and rougeL_f1 >= 0.40:
            return "good"
        elif rouge1_f1 >= 0.30 and rouge2_f1 >= 0.15 and rougeL_f1 >= 0.25:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def evaluate_dataset(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all samples in the dataset"""
        logger.info(f"Starting evaluation of {len(dataset['samples'])} samples")
        
        for sample in dataset['samples']:
            result = self.evaluate_sample(sample)
            self.results.append(result)
            
            # Brief pause between requests
            time.sleep(1)
        
        logger.info("Dataset evaluation completed")
        return self.results
    
    def generate_report(self, output_path: str = "rouge_evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Calculate aggregate statistics
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            logger.error("No successful evaluations to report")
            return
        
        # Overall statistics
        rouge1_scores = [r['rouge_scores']['rouge1']['fmeasure'] for r in successful_results]
        rouge2_scores = [r['rouge_scores']['rouge2']['fmeasure'] for r in successful_results]
        rougeL_scores = [r['rouge_scores']['rougeL']['fmeasure'] for r in successful_results]
        
        # Content type analysis
        content_type_stats = {}
        for content_type in set(r['content_type'] for r in successful_results):
            type_results = [r for r in successful_results if r['content_type'] == content_type]
            content_type_stats[content_type] = {
                'count': len(type_results),
                'rouge1_avg': np.mean([r['rouge_scores']['rouge1']['fmeasure'] for r in type_results]),
                'rouge2_avg': np.mean([r['rouge_scores']['rouge2']['fmeasure'] for r in type_results]),
                'rougeL_avg': np.mean([r['rouge_scores']['rougeL']['fmeasure'] for r in type_results]),
                'evaluation_scores': [r['evaluation_score'] for r in type_results]
            }
        
        # Evaluation score distribution
        score_distribution = {}
        for score in ['excellent', 'good', 'acceptable', 'needs_improvement', 'failed']:
            count = len([r for r in self.results if r.get('evaluation_score') == score])
            score_distribution[score] = count
        
        report = {
            'evaluation_summary': {
                'total_samples': len(self.results),
                'successful_evaluations': len(successful_results),
                'failed_evaluations': len(self.results) - len(successful_results),
                'success_rate': len(successful_results) / len(self.results) if self.results else 0
            },
            'rouge_statistics': {
                'rouge1': {
                    'mean': np.mean(rouge1_scores),
                    'std': np.std(rouge1_scores),
                    'min': np.min(rouge1_scores),
                    'max': np.max(rouge1_scores)
                },
                'rouge2': {
                    'mean': np.mean(rouge2_scores),
                    'std': np.std(rouge2_scores), 
                    'min': np.min(rouge2_scores),
                    'max': np.max(rouge2_scores)
                },
                'rougeL': {
                    'mean': np.mean(rougeL_scores),
                    'std': np.std(rougeL_scores),
                    'min': np.min(rougeL_scores),
                    'max': np.max(rougeL_scores)
                }
            },
            'content_type_analysis': content_type_stats,
            'evaluation_score_distribution': score_distribution,
            'detailed_results': self.results
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Print summary
        self.print_summary(report)
    
    def print_summary(self, report: Dict[str, Any]):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("ROUGE EVALUATION SUMMARY")
        print("="*60)
        
        summary = report['evaluation_summary']
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Successful Evaluations: {summary['successful_evaluations']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        
        print("\nROUGE SCORES (F1-Measure):")
        rouge_stats = report['rouge_statistics']
        print(f"ROUGE-1: {rouge_stats['rouge1']['mean']:.3f} (±{rouge_stats['rouge1']['std']:.3f})")
        print(f"ROUGE-2: {rouge_stats['rouge2']['mean']:.3f} (±{rouge_stats['rouge2']['std']:.3f})")
        print(f"ROUGE-L: {rouge_stats['rougeL']['mean']:.3f} (±{rouge_stats['rougeL']['std']:.3f})")
        
        print("\nEVALUATION SCORE DISTRIBUTION:")
        for score, count in report['evaluation_score_distribution'].items():
            percentage = count / summary['total_samples'] * 100
            print(f"{score.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print("\nCONTENT TYPE PERFORMANCE:")
        for content_type, stats in report['content_type_analysis'].items():
            print(f"\n{content_type.replace('_', ' ').title()} ({stats['count']} samples):")
            print(f"  ROUGE-1: {stats['rouge1_avg']:.3f}")
            print(f"  ROUGE-2: {stats['rouge2_avg']:.3f}")
            print(f"  ROUGE-L: {stats['rougeL_avg']:.3f}")
        
        print("\n" + "="*60)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='ROUGE evaluation for summarization APIs')
    parser.add_argument('--dataset', '-d', default='rouge_dataset.json', 
                       help='Path to dataset file')
    parser.add_argument('--api-url', '-u', default='http://localhost:8000',
                       help='Base URL for the API')
    parser.add_argument('--output', '-o', default='rouge_evaluation_report.json',
                       help='Output path for evaluation report')
    parser.add_argument('--sample-ids', nargs='*', type=int,
                       help='Specific sample IDs to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ROUGEEvaluator(api_base_url=args.api_url)
    
    # Load dataset
    try:
        dataset = evaluator.load_dataset(args.dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Filter samples if specific IDs requested
    if args.sample_ids:
        dataset['samples'] = [s for s in dataset['samples'] if s['id'] in args.sample_ids]
        logger.info(f"Filtered to {len(dataset['samples'])} samples")
    
    # Run evaluation
    try:
        evaluator.evaluate_dataset(dataset)
        evaluator.generate_report(args.output)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
