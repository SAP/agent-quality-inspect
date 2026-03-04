import pytest

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

from agent_inspect.models.tools import ErrorAnalysisDataSample, ErrorAnalysisResult
from agent_inspect.tools import ErrorAnalysis

from tests_acceptance.azure_openai_client import AzureOpenAIClient
from tests_acceptance.tools.scripts.utils import load_error_analysis_data, select_n_agent_runs

PATH_TO_ERROR_SUMMARY_CSV = "tests_acceptance/tools/sample_data/final_error_summary.csv"
NUMBER_OF_AGENT_RUNS = 10

@pytest.fixture
def error_analysis_data_samples() -> List[ErrorAnalysisDataSample]:
    """Fixture to load error analysis data samples from CSV for testing."""
    data_samples = load_error_analysis_data(PATH_TO_ERROR_SUMMARY_CSV, NUMBER_OF_AGENT_RUNS)
    return data_samples

@pytest.fixture
def ground_truth_error_categories() -> List[str]:
    """Fixture to load ground truth error categories from pre-run error analysis."""
    ground_truth_rows = select_n_agent_runs(pd.read_csv(PATH_TO_ERROR_SUMMARY_CSV), NUMBER_OF_AGENT_RUNS)
    print(f"Loaded", len(ground_truth_rows), "ground truth rows.")

    ground_truth_cluster_col = ground_truth_rows[['cluster_label']]
    ground_truth_clusters = ground_truth_cluster_col['cluster_label'].dropna().unique().tolist()
    return ground_truth_clusters

@pytest.fixture
def azure_openai_client():
    return AzureOpenAIClient(model="gpt-4.1", max_tokens=16384, temperature=1)
    
def normalize_category(category: str) -> str:
    """Normalize category string for better matching"""
    if pd.isna(category):
        return ""
    # Convert to lowercase, remove extra spaces
    normalized = category.lower().strip()
    # Standardize separators
    normalized = re.sub(r'[:/\-]+', ' ', normalized)
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def extract_category_patterns(category: str) -> Dict[str, Any]:
    """
    Extract semantic patterns from error category.
    Returns action type, target entity, and key terms.
    """
    normalized = normalize_category(category)
    
    # Extract action type (first major word/phrase)
    action_patterns = {
        'missing': ['missing', 'not attempted', 'blocked'],
        'incorrect': ['incorrect', 'wrong', 'invalid', 'improper'],
        'incomplete': ['incomplete', 'partial'],
    }
    
    action_type = 'other'
    for action, keywords in action_patterns.items():
        if any(kw in normalized for kw in keywords):
            action_type = action
            break
    
    # Extract target entity (what's being operated on)
    entity_patterns = {
        'tool_usage': ['tool usage', 'tool', 'function', 'api'],
        'communication': ['communication', 'output', 'message', 'inform'],
        'parameter': ['parameter', 'argument', 'input', 'value'],
        'reservation': ['reservation', 'booking', 'book'],
        'payment': ['payment', 'charge', 'amount', 'price', 'cost'],
        'baggage': ['baggage', 'luggage', 'bag'],
        'flight': ['flight', 'ticket'],
        'passenger': ['passenger', 'customer'],
        'certificate': ['certificate', 'voucher'],
        'upgrade': ['upgrade', 'upgraded'],
        'cancel': ['cancel', 'cancellation'],
        'update': ['update', 'modify', 'change'],
        'calculation': ['calculation', 'compute', 'aggregate'],
        'policy': ['policy', 'rule', 'membership'],
    }
    
    entities = set()
    for entity, keywords in entity_patterns.items():
        if any(kw in normalized for kw in keywords):
            entities.add(entity)
    
    # Extract key terms (nouns/verbs excluding common words)
    stopwords = {'the', 'a', 'an', 'in', 'or', 'and', 'of', 'to', 'for', 'with', 'on', 'at', 'from', 'by'}
    words = normalized.split()
    key_terms = set(w for w in words if len(w) > 3 and w not in stopwords)
    
    return {
        'original': category,
        'normalized': normalized,
        'action_type': action_type,
        'entities': entities,
        'key_terms': key_terms
    }

def calculate_pattern_similarity(pattern1: Dict, pattern2: Dict) -> float:
    """
    Calculate similarity between two category patterns.
    Uses multiple signals: action type, entities, and term overlap.
    """
    # Action type match (binary: 0 or 1)
    action_match = 1.0 if pattern1['action_type'] == pattern2['action_type'] else 0.0
    
    # Entity overlap (Jaccard)
    entities1 = pattern1['entities']
    entities2 = pattern2['entities']
    if entities1 or entities2:
        entity_intersection = entities1.intersection(entities2)
        entity_union = entities1.union(entities2)
        entity_overlap = len(entity_intersection) / len(entity_union) if entity_union else 0.0
    else:
        entity_overlap = 0.0
    
    # Key term overlap (Jaccard)
    terms1 = pattern1['key_terms']
    terms2 = pattern2['key_terms']
    if terms1 or terms2:
        term_intersection = terms1.intersection(terms2)
        term_union = terms1.union(terms2)
        term_overlap = len(term_intersection) / len(term_union) if term_union else 0.0
    else:
        term_overlap = 0.0
    
    # Weighted combination:
    # - 30% action type (critical for category type)
    # - 35% entity overlap (what's being worked on)
    # - 35% term overlap (semantic content)
    similarity = (
        0.30 * action_match +
        0.35 * entity_overlap +
        0.35 * term_overlap
    )
    
    return similarity

def find_best_pattern_match(target_pattern: Dict, candidate_patterns: List[Dict]) -> Tuple[Dict, float]:
    """
    Find best matching pattern from candidates using pattern similarity.
    Returns: (best_match_pattern, similarity_score)
    """
    if not candidate_patterns:
        return None, 0.0
    
    best_match = None
    best_score = -1
    
    for candidate in candidate_patterns:
        score = calculate_pattern_similarity(target_pattern, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match, best_score

def compare_categories(oracle_categories: List[str], 
                       test_categories: List[str],
                       threshold: float = 0.6) -> Dict:
    """
    Hierarchical pattern-based matching with semantic understanding.
    
    The approach:
    1. Extracts semantic patterns (action type, entities, key terms)
    2. Uses weighted multi-signal similarity
    3. Performs bidirectional matching
    4. Provides detailed diagnostics
    
    Args:
        oracle_categories: Reference error categories
        test_categories: Test run error categories
        threshold: Minimum similarity for a good match (default: 0.6)
    
    Returns:
        Dict with results, matches, and pass/fail verdict
    """
    # Extract patterns
    oracle_patterns = [extract_category_patterns(cat) for cat in oracle_categories]
    test_patterns = [extract_category_patterns(cat) for cat in test_categories]
    
    # Forward matching: oracle -> test
    forward_matches = []
    for o_pattern in oracle_patterns:
        best_match, score = find_best_pattern_match(o_pattern, test_patterns)
        forward_matches.append({
            'oracle': o_pattern['original'],
            'test_match': best_match['original'] if best_match else None,
            'similarity': score,
            'oracle_action': o_pattern['action_type'],
            'oracle_entities': o_pattern['entities'],
            'match_action': best_match['action_type'] if best_match else None,
            'match_entities': best_match['entities'] if best_match else set(),
        })
    
    # Backward matching: test -> oracle
    backward_matches = []
    for t_pattern in test_patterns:
        best_match, score = find_best_pattern_match(t_pattern, oracle_patterns)
        backward_matches.append({
            'test': t_pattern['original'],
            'oracle_match': best_match['original'] if best_match else None,
            'similarity': score,
            'test_action': t_pattern['action_type'],
            'test_entities': t_pattern['entities'],
            'match_action': best_match['action_type'] if best_match else None,
            'match_entities': best_match['entities'] if best_match else set(),
        })
    
    # Calculate metrics
    forward_scores = [m['similarity'] for m in forward_matches]
    backward_scores = [m['similarity'] for m in backward_matches]
    
    avg_forward = np.mean(forward_scores) if forward_scores else 0.0
    avg_backward = np.mean(backward_scores) if backward_scores else 0.0
    bidirectional_avg = (avg_forward + avg_backward) / 2
    
    # Count matches above threshold
    good_forward_matches = sum(1 for s in forward_scores if s >= threshold)
    good_backward_matches = sum(1 for s in backward_scores if s >= threshold)
    
    # Precision and Recall-like metrics
    precision = good_forward_matches / len(forward_scores) if forward_scores else 0.0
    recall = good_backward_matches / len(backward_scores) if backward_scores else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'forward_matches': forward_matches,
        'backward_matches': backward_matches,
        'avg_forward_similarity': avg_forward,
        'avg_backward_similarity': avg_backward,
        'bidirectional_avg': bidirectional_avg,
        'good_forward_matches': good_forward_matches,
        'good_backward_matches': good_backward_matches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'num_oracle': len(oracle_categories),
        'num_test': len(test_categories),
        'threshold': threshold
    }

def print_results(results: Dict, verbose: bool = True) -> None:
    """Print detailed results from Approach 3"""
    print("=" * 80)
    print("🎯 APPROACH 3: HIERARCHICAL PATTERN-BASED MATCHING")
    print("=" * 80)
    
    print(f"\n📊 Dataset Size:")
    print(f"   Oracle categories: {results['num_oracle']}")
    print(f"   Test categories:   {results['num_test']}")
    
    print(f"\n📈 Similarity Scores:")
    print(f"   Oracle → Test (Forward):  {results['avg_forward_similarity']:.3f}")
    print(f"   Test → Oracle (Backward): {results['avg_backward_similarity']:.3f}")
    print(f"   Bidirectional Average:    {results['bidirectional_avg']:.3f}")
    
    print(f"\n🎯 Match Quality (threshold = {results['threshold']}):")
    print(f"   Good Forward Matches:  {results['good_forward_matches']}/{results['num_oracle']} ({results['precision']*100:.1f}%)")
    print(f"   Good Backward Matches: {results['good_backward_matches']}/{results['num_test']} ({results['recall']*100:.1f}%)")
    print(f"   F1 Score: {results['f1_score']:.3f}")
    
    # Determine pass/fail
    # Use F1 score as primary metric (balances precision and recall)
    passed = results['f1_score'] >= results['threshold']
    
    print(f"\n{'✅ TEST PASSED' if passed else '❌ TEST FAILED'}")
    print(f"   F1 Score: {results['f1_score']:.3f} {'≥' if passed else '<'} Threshold: {results['threshold']}")
    
    if verbose:
        # Show some example matches
        print(f"\n🔍 Top Matches (sorted by similarity):")
        sorted_forward = sorted(results['forward_matches'], key=lambda x: x['similarity'], reverse=True)
        for i, match in enumerate(sorted_forward[:5], 1):
            print(f"\n   {i}. Similarity: {match['similarity']:.3f}")
            print(f"      Oracle: {match['oracle'][:70]}")
            print(f"      Test:   {match['test_match'][:70] if match['test_match'] else 'None'}")
            print(f"      Action: {match['oracle_action']} → {match['match_action']}")
            print(f"      Entities: {match['oracle_entities']} → {match['match_entities']}")
        
        print(f"\n⚠️  Poor Matches (similarity < {results['threshold']}):")
        poor_matches = [m for m in results['forward_matches'] if m['similarity'] < results['threshold']]
        if poor_matches:
            for i, match in enumerate(poor_matches[:5], 1):
                print(f"\n   {i}. Similarity: {match['similarity']:.3f}")
                print(f"      Oracle: {match['oracle'][:70]}")
                print(f"      Best Test Match: {match['test_match'][:70] if match['test_match'] else 'None'}")
        else:
            print("   None - All matches above threshold! ✨")
    
    print("\n" + "=" * 80)

def test_error_analysis_accuracy(error_analysis_data_samples, ground_truth_error_categories, azure_openai_client):
    """Test error analysis accuracy using hierarchical pattern-based matching."""
    F1_SCORE_THRESHOLD = 0.55

    # Perform error analysis on sample data
    error_analyzer = ErrorAnalysis(llm_client=azure_openai_client)
    test_error_analysis_results: ErrorAnalysisResult = error_analyzer.analyze_batch(error_analysis_data_samples)
    test_error_categories = list(test_error_analysis_results.analyzed_validations_clustered_by_errors.keys())
    
    # Run comparison
    results = compare_categories(ground_truth_error_categories, test_error_categories, threshold=F1_SCORE_THRESHOLD)
    print_results(results, verbose=True)

    assert results["f1_score"] >= F1_SCORE_THRESHOLD, f"F1 Score {results['f1_score']:.3f} below threshold {F1_SCORE_THRESHOLD}"
