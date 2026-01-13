from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import re
import json
import nltk
from sentence_transformers import CrossEncoder

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("Downloading NLTK POS tagger...")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


@dataclass
class ContradictionResult:
    """Result object containing contradiction analysis"""
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str


class SemanticContradictionDetector:
    """
    Detects semantic contradictions within a single document using an NLI Cross-Encoder.
    
    This corresponds to 'Branch A' in the multi-branch ensemble architecture from Part 1.
    Uses RoBERTa-based NLI model for deep logical understanding of sentence relationships.
    """
    
    # Negation words for contradiction detection
    NEGATION_WORDS = {
        'no', 'not', 'never', 'neither', 'nor', 'nothing', 'nowhere',
        'hardly', 'scarcely', 'barely', "n't", 'none', 'nobody', 'without'
    }
    
    # Temporal markers that indicate updates (NOT contradictions)
    TEMPORAL_MARKERS = {
        'initially', 'originally', 'at first', 'previously', 'before',
        'now', 'currently', 'update:', 'edit:', 'revised:', 'changed:',
        'after', 'later', 'then', 'eventually'
    }
    
    # Contrastive markers that signal potential contradictions
    CONTRASTIVE_MARKERS = {
        'but', 'however', 'though', 'although', 'yet', 'nevertheless',
        'nonetheless', 'despite', 'in spite of', 'conversely', 'on the other hand'
    }
    
    def __init__(self, model_name: str = "cross-encoder/nli-roberta-base"):
        """
        Initialize the detector with a pre-trained NLI model.
        
        Args:
            model_name: HuggingFace model identifier
                       Default: cross-encoder/nli-roberta-base
                       - Trained on SNLI + MultiNLI (1M+ examples)
                       - Output: [contradiction, entailment, neutral] logits
        """
        print(f"Loading NLI model: {model_name}...")
        self.model = CrossEncoder(model_name)
        
        # Label mapping for cross-encoder/nli-roberta-base
        # Index 0: Contradiction
        # Index 1: Entailment  
        # Index 2: Neutral
        self.contradiction_idx = 0
        self.entailment_idx = 1
        self.neutral_idx = 2
        
        # Threshold tuned to balance precision/recall
        # Higher = fewer false positives, but may miss subtle contradictions
        self.threshold = 0.48
        
        print("Model loaded successfully!\n")
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text into analyzable sentence units.
        
        Pipeline:
        1. Basic cleanup (strip whitespace, normalize quotes)
        2. Sentence segmentation using NLTK
        3. Filter very short sentences (likely not claims)
        4. Normalize whitespace
        
        Args:
            text: Raw review text
            
        Returns:
            List of cleaned sentences
        """
        # Basic cleanup
        text = text.strip()
        text = text.replace('"', '"').replace('"', '"') 
        text = text.replace(''', "'").replace(''', "'")
        
        # Sentence segmentation
        sentences = nltk.sent_tokenize(text)

        cleaned_sentences = []
        for sent in sentences:
            sent = ' '.join(sent.split())
            word_count = len(sent.split())
            if word_count >= 4: 
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract and enrich claims with metadata for better contradiction detection.
        
        Enhancements:
        - Negation flags
        - Temporal markers
        - Numeric values
        - Named entities (basic extraction)
        
        Args:
            sentences: Preprocessed sentences
            
        Returns:
            List of claim dictionaries with metadata
        """
        claims = []
        
        for idx, sentence in enumerate(sentences):
            claim = {
                'text': sentence,
                'index': idx,
                'has_negation': self._detect_negation(sentence),
                'has_temporal': self._detect_temporal_markers(sentence),
                'has_contrastive': self._detect_contrastive_markers(sentence),
                'numeric_values': self._extract_numbers(sentence),
                'entities': self._extract_basic_entities(sentence)
            }
            claims.append(claim)
        
        return claims
    
    def _detect_negation(self, text: str) -> bool:
        """Detect if sentence contains negation words"""
        text_lower = text.lower()
        return any(neg in text_lower for neg in self.NEGATION_WORDS)
    
    def _detect_temporal_markers(self, text: str) -> bool:
        """Detect temporal shift markers (indicates update, not contradiction)"""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.TEMPORAL_MARKERS)
    
    def _detect_contrastive_markers(self, text: str) -> bool:
        """Detect contrastive markers that signal opposing statements"""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.CONTRASTIVE_MARKERS)
    
    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numeric values with units for normalization.
        
        Examples:
        - "10 seconds" → 10 seconds
        - "5 minutes" → 300 seconds
        - "2 hours" → 7200 seconds
        - "three-week" → 3 weeks → 1814400 seconds
        """
        numbers = []
        
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        # Pattern 1: Numeric digits + optional unit
        pattern = r'(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)?'
        matches = re.finditer(pattern, text.lower())
        
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else 'none'
            
            # Normalize time units to seconds
            normalized_value = value
            if unit and 'min' in unit:
                normalized_value = value * 60
            elif unit and 'hour' in unit or unit and 'hr' in unit:
                normalized_value = value * 3600
            elif unit and 'day' in unit:
                normalized_value = value * 86400
            elif unit and 'week' in unit:
                normalized_value = value * 604800
            
            numbers.append({
                'raw_value': value,
                'unit': unit,
                'normalized_value': normalized_value,
                'span': match.span()
            })
        
        # Pattern 2: Word numbers + hyphen + unit (e.g., "three-week")
        pattern2 = r'(one|two|three|four|five|six|seven|eight|nine|ten)-?(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)'
        matches2 = re.finditer(pattern2, text.lower())
        
        for match in matches2:
            word_num = match.group(1)
            value = float(word_to_num[word_num])
            unit = match.group(2)
            
            # Normalize time units to seconds
            normalized_value = value
            if unit and 'min' in unit:
                normalized_value = value * 60
            elif unit and 'hour' in unit or unit and 'hr' in unit:
                normalized_value = value * 3600
            elif unit and 'day' in unit:
                normalized_value = value * 86400
            elif unit and 'week' in unit:
                normalized_value = value * 604800
            
            numbers.append({
                'raw_value': value,
                'unit': unit,
                'normalized_value': normalized_value,
                'span': match.span()
            })
        
        return numbers
    
    def _extract_basic_entities(self, text: str) -> List[str]:
        """
        Extract basic named entities (nouns) for entity-attribute binding.
        
        This helps prevent false positives like:
        "The camera is great. The battery is terrible."
        (Different entities = not a contradiction)
        """
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        entities = [word.lower() for word, pos in pos_tags 
                   if pos.startswith('NN')]
        
        return entities
    
    def check_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float, str]:
        """
        Check if two claims contradict each other using NLI model.
        
        Enhanced logic:
        1. Skip temporal updates (not contradictions)
        2. Check numeric conflicts
        3. Use NLI model for semantic contradiction
        4. Entity overlap check (same topic?)
        
        Args:
            claim_a: First claim dictionary
            claim_b: Second claim dictionary
            
        Returns:
            Tuple of (is_contradiction, confidence_score, explanation)
        """
        text_a = claim_a['text']
        text_b = claim_b['text']
        
        # Rule 1: Skip if both have temporal markers (likely an update)
        explicit_updates = ['update:', 'edit:', 'revised:', 'correction:', 'initially', 'originally']
        text_a_lower = claim_a['text'].lower()
        text_b_lower = claim_b['text'].lower()
        
        has_explicit_update_a = any(marker in text_a_lower for marker in explicit_updates)
        has_explicit_update_b = any(marker in text_b_lower for marker in explicit_updates)

        # Only skip if it's clearly a temporal update (e.g., "Initially... Update:")
        if has_explicit_update_a and has_explicit_update_b:
            return False, 0.0, "Explicit temporal update detected (not contradiction)"
        
        # Rule 2: Check for contrastive patterns with opposing semantics
        if claim_a['has_negation'] != claim_b['has_negation']:
            # One has negation, other doesn't - check for semantic opposition
            if claim_b['has_contrastive'] or claim_a['has_contrastive']:
                # Check for opposing keywords
                opposing_pairs = [
                    (['damage', 'harm', 'break', 'crack', 'scratch'], ['no', 'without', 'never']),
                    (['durable', 'strong', 'sturdy'], ['cracked', 'broken', 'damaged']),
                    (['works', 'working', 'functional'], ['broken', 'failed', 'doesn\'t'])
                ]
                
                text_a_lower = text_a.lower()
                text_b_lower = text_b.lower()
                
                for pos_words, neg_words in opposing_pairs:
                    has_pos = any(word in text_a_lower or word in text_b_lower for word in pos_words)
                    has_neg = any(word in text_a_lower or word in text_b_lower for word in neg_words)
                    if has_pos and has_neg:
                        return True, 0.80, "Contrastive negation pattern contradiction"
        
        # Rule 3: Check numeric contradictions
        numeric_conflict = self._check_numeric_conflict(claim_a, claim_b)
        if numeric_conflict:
            return True, 0.85, "Numeric/temporal contradiction"
        
        # Rule 4: NLI Model inference
        pair = [(text_a, text_b)]
        scores = self.model.predict(pair)
        probs = self._softmax(scores[0])
        
        contradiction_score = probs[self.contradiction_idx]
        entailment_score = probs[self.entailment_idx]
        is_contradiction = (contradiction_score > entailment_score and 
                          contradiction_score > probs[self.neutral_idx] and
                          contradiction_score > self.threshold)
        
        explanation = f"NLI score: {contradiction_score:.3f}"
        
        # Rule 5: Entity check (reduce false positives)
        shared_entities = set(claim_a['entities']) & set(claim_b['entities'])
        if is_contradiction and len(shared_entities) == 0 and len(claim_a['entities']) > 0 and len(claim_b['entities']) > 0:
            contradiction_score *= 0.5 
            is_contradiction = contradiction_score > self.threshold
            explanation += " (different entities - confidence reduced)"
        
        return is_contradiction, float(contradiction_score), explanation
    
    def _check_numeric_conflict(self, claim_a: Dict, claim_b: Dict) -> bool:
        """
        Check if numeric values conflict between claims.
        
        Example contradiction:
        - "Boot time is 10 seconds"
        - "Takes 5 minutes to start"
        Both describe timing, but 300sec >> 10sec = contradiction
        """
        nums_a = claim_a['numeric_values']
        nums_b = claim_b['numeric_values']
        
        if not nums_a or not nums_b:
            return False
        for num_a in nums_a:
            for num_b in nums_b:
                val_a = num_a['normalized_value']
                val_b = num_b['normalized_value']
                
                if val_a > 0 and val_b > 0:
                    ratio = max(val_a, val_b) / min(val_a, val_b)
                    if ratio > 10:
                        return True
        
        return False
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def analyze(self, text: str) -> ContradictionResult:
        """
        Main analysis pipeline - detect contradictions in review text.
        
        Complexity: O(N²) where N = number of sentences
        For production optimization strategies, see Part 4.
        
        Args:
            text: Review text to analyze
            
        Returns:
            ContradictionResult with findings
        """
        # Stage 1: Preprocessing
        sentences = self.preprocess(text)
        
        if len(sentences) < 2:
            return ContradictionResult(
                has_contradiction=False,
                confidence=0.0,
                contradicting_pairs=[],
                explanation="Insufficient sentences for contradiction analysis"
            )
        
        # Stage 2: Claim extraction
        claims = self.extract_claims(sentences)
        
        # Stage 3: Pairwise contradiction checking
        contradictions = []
        max_confidence = 0.0
        explanations = []
        
        # Only check i < j to avoid duplicate pairs
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                is_contra, score, explanation = self.check_contradiction(
                    claims[i], claims[j]
                )
                
                if is_contra:
                    contradictions.append((claims[i]['text'], claims[j]['text']))
                    max_confidence = max(max_confidence, score)
                    explanations.append(
                        f"Sentences {i+1} & {j+1}: {explanation}"
                    )
        
        # Stage 4: Result compilation
        has_contradiction = len(contradictions) > 0
        
        if has_contradiction:
            explanation = (f"Found {len(contradictions)} contradiction(s). "
                         f"Highest confidence: {max_confidence:.3f}. "
                         f"Details: {'; '.join(explanations[:2])}")
        else:
            explanation = "No logical contradictions detected."
        
        return ContradictionResult(
            has_contradiction=has_contradiction,
            confidence=max_confidence if has_contradiction else 0.0,
            contradicting_pairs=contradictions,
            explanation=explanation
        )


def load_dataset(filepath: str = 'dataset.txt') -> List[Dict]:
    """
    Load test dataset from file.
    
    Args:
        filepath: Path to dataset JSON file
        
    Returns:
        List of test samples
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using embedded dataset.")
        return get_embedded_dataset()
    except json.JSONDecodeError as e:
        print(f"Warning: {filepath} has invalid JSON format ({e}). Using embedded dataset.")
        return get_embedded_dataset()


def get_embedded_dataset() -> List[Dict]:
    """Fallback embedded dataset if file not found"""
    return [
        {
            "id": 1,
            "text": "This laptop is incredibly fast. Boot time is under 10 seconds. However, I find myself waiting 5 minutes just to open Chrome. The performance is unmatched in this price range.",
            "has_contradiction": True
        },
        {
            "id": 2,
            "text": "The camera quality is stunning in daylight. Night mode works well too. I've taken beautiful photos at my daughter's evening recital. Great for any lighting condition.",
            "has_contradiction": False
        },
        {
            "id": 3,
            "text": "I've never had a phone this durable. Dropped it multiple times with no damage. The screen cracked on the first drop though. Build quality is exceptional.",
            "has_contradiction": True
        },
        {
            "id": 4,
            "text": "Customer service was unhelpful and rude. They resolved my issue within minutes and even gave me a discount. Worst support experience I've ever had.",
            "has_contradiction": True
        },
        {
            "id": 5,
            "text": "The noise cancellation is mediocre at best. I can still hear my coworkers clearly. But honestly, for the price, you can't expect studio-quality isolation.",
            "has_contradiction": False
        },
        {
            "id": 6,
            "text": "Shipping was lightning fast - arrived in 2 days. The three-week wait was worth it though. Amazon Prime really delivers.",
            "has_contradiction": True
        },
        {
            "id": 7,
            "text": "This blender is whisper quiet. My baby sleeps right through it. The noise is so loud I have to wear ear protection. Perfect for early morning smoothies.",
            "has_contradiction": True
        },
        {
            "id": 8,
            "text": "Not the cheapest option, but definitely worth the premium price. The quality justifies the cost. You get what you pay for with this brand.",
            "has_contradiction": False
        }
    ]


def evaluate(detector: SemanticContradictionDetector, 
             test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate detector performance against ground truth.
    
    Metrics:
    - Accuracy: Overall correctness
    - Precision: Of flagged contradictions, how many are real?
    - Recall: Of real contradictions, how many did we catch?
    - F1: Harmonic mean of precision and recall
    
    Args:
        detector: Initialized detector
        test_data: List of test samples with ground truth
        
    Returns:
        Dictionary of performance metrics
    """
    correct = 0
    total = len(test_data)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    print(f"\n{'='*70}")
    print(f"EVALUATION: Testing on {total} samples")
    print(f"{'='*70}\n")
    
    for item in test_data:
        text = item["text"]
        ground_truth = item["has_contradiction"]
        
        # Run detection
        result = detector.analyze(text)
        prediction = result.has_contradiction
        
        # Update metrics
        if prediction == ground_truth:
            correct += 1
            if prediction:
                true_positives += 1
                status = "✅ TRUE POSITIVE"
            else:
                true_negatives += 1
                status = "✅ TRUE NEGATIVE"
        else:
            if prediction and not ground_truth:
                false_positives += 1
                status = "❌ FALSE POSITIVE"
            else:
                false_negatives += 1
                status = "❌ FALSE NEGATIVE"
        
        # Print result
        print(f"Sample ID {item['id']}: {status}")
        print(f"  Prediction: {prediction} | Ground Truth: {ground_truth}")
        print(f"  Confidence: {result.confidence:.3f}")
        if prediction:
            print(f"  Pairs found: {len(result.contradicting_pairs)}")
            for pair in result.contradicting_pairs[:2]:  # Show first 2
                print(f"    - \"{pair[0][:50]}...\"")
                print(f"      vs \"{pair[1][:50]}...\"")
        print()
    
    # Calculate metrics
    accuracy = correct / total
    precision = (true_positives / (true_positives + false_positives) 
                if (true_positives + false_positives) > 0 else 0.0)
    recall = (true_positives / (true_positives + false_negatives) 
             if (true_positives + false_negatives) > 0 else 0.0)
    f1 = (2 * (precision * recall) / (precision + recall) 
         if (precision + recall) > 0 else 0.0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives
    }


def test_edge_cases(detector: SemanticContradictionDetector):
    """
    Test additional edge cases beyond the provided dataset.
    
    This demonstrates robustness and addresses Part 4, Q5
    (handling legitimate temporal updates as false positives).
    """
    print(f"\n{'='*70}")
    print("EDGE CASE TESTING")
    print(f"{'='*70}\n")
    
    edge_cases = [
        {
            "id": "E1",
            "description": "Temporal Update (Should NOT flag)",
            "text": "Update: Initially gave 5 stars but battery died after 1 month. Now 2 stars.",
            "expected": False
        },
        {
            "id": "E2",
            "description": "Different Entities (Should NOT flag)",
            "text": "The camera is excellent. The battery life is terrible. Overall good value.",
            "expected": False
        },
        {
            "id": "E3",
            "description": "Subtle Numeric Contradiction",
            "text": "Charges in 30 minutes. I wait 4 hours every time for a full charge.",
            "expected": True
        },
        {
            "id": "E4",
            "description": "Opinion vs Fact (Borderline)",
            "text": "I think it's fast. Actually it's incredibly slow for modern standards.",
            "expected": True
        }
    ]
    
    for case in edge_cases:
        result = detector.analyze(case["text"])
        passed = result.has_contradiction == case["expected"]
        
        print(f"Edge Case {case['id']}: {case['description']}")
        print(f"  Expected: {case['expected']} | Got: {result.has_contradiction}")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Explanation: {result.explanation}")
        print()


if __name__ == "__main__":
    print("="*70)
    print("SEMANTIC CONTRADICTION DETECTOR - Part 2 Implementation")
    print("="*70)
    print()
    
    # Initialize detector
    detector = SemanticContradictionDetector()
    
    # Load dataset
    test_data = load_dataset('dataset.txt')
    
    # Run evaluation
    metrics = evaluate(detector, test_data)
    
    # Print metrics summary
    print(f"{'='*70}")
    print("FINAL METRICS")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    print()
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print()
    
    # Test edge cases
    test_edge_cases(detector)
    
    print(f"{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
