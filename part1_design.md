# Part 1: NLP Pipeline Design Document

## 1. Architecture Overview

We utilize a **Multi Branch Ensemble Architecture**. The system processes reviews in parallel streams to detect three distinct types of deception signals: logical (contradictions), psychological (manipulation), and stylistic (identity fraud).

```
                                  [ INPUT REVIEW ]
                                         |
                                         v
                            +--------------------------+
                            |   DUAL PREPROCESSING     |
                            | Stream 1: Normalized     |
                            | Stream 2: Raw (Style)    |
                            +--------------------------+
                                         |
            +----------------------------+----------------------------+
            |                            |                            |
            v                            v                            v
  +-------------------+        +-------------------+        +--------------------+
  |     BRANCH A      |        |     BRANCH B      |        |     BRANCH C       |
  |   Contradiction   |        |   Manipulation    |        |   Fingerprinting   |
  |     (Logic)       |        |     (Intent)      |        |     (Identity)     |
  +-------------------+        +-------------------+        +--------------------+
  | 1. Split Sentences|        | 1. Urgency/FOMO   |        | 1. Extract n-grams |
  |                   |        |    Extraction     |        |    & Function Words|
  | 2. Extract Claims |        |                   |        |                    |
  |    (dates/nums)   |        | 2. Sentiment      |        | 2. Vectorize       |
  |                   |        |    Volatility     |        |    (Style Vector)  |
  | 3. Cross-Encoder  |        |                   |        |                    |
  |    (RoBERTa-MNLI) |        | 3. Intent Model   |        | 3. Distance Check  |
  |                   |        |    (DistilBERT)   |        |    (Cosine/Delta)  |
  +-------------------+        +-------------------+        +--------------------+
            |                            |                            |
            v                            v                            v
      [ Logic Score ]              [ Psych Score ]            [ Style Score ]
        (0.0 - 1.0)                  (0.0 - 1.0)              (Anomaly Metric)
            |                            |                            |
            +----------------------------+----------------------------+
                                         |
                                         v
                            +--------------------------+
                            |    AGGREGATION ENGINE    |
                            | - Weighted Voting        |
                            | - Threshold Checks       |
                            | - Confidence Calibration |
                            +--------------------------+
                                         |
                                         v
                                [ FINAL VERDICT ]
```

---

## 2. Preprocessing Pipeline

We employ a **Dual-Stream Preprocessing** strategy because different branches require different data representations:

### Stream 1 (Semantic Analysis - Branches A & B)

- **Normalization**: Lowercasing, removing HTML tags and special characters
- **Language Detection**: Filter non English reviews using fastText (models are English specific)
- **Segmentation**: Utilizing spaCy for sentence boundary detection (critical for pair generation)
- **Coreference Resolution**: Resolving pronouns like "it" to "the battery" to ensure contradictions are caught contextually

### Stream 2 (Stylometry - Branch C)

- **Raw Retention**: Preserves capitalization, punctuation, and whitespace (e.g., double spacing) as these are key authorial fingerprints

---

## 3. Feature Engineering Strategy

We extract distinct feature sets for each branch to target specific deception vectors:

### Branch A: Semantic Contradiction Signals

- **Sentence Pairs**: Generate all possible intra review sentence pairs (N × N-1 combinations)
- **Claim Extraction**: Isolate verifiable factual statements (e.g., "charges twice daily" = testable claim) to focus NLI on contradictable content rather than subjective opinions
- **Numeric/Temporal Normalization**: Convert relative time ("after a month") and units ("10 seconds", "5 minutes") into standardized values to detect factual conflicts
- **Negation Detection**: Flag explicit negators (no, not, never, hardly) which often pivot the meaning of a sentence
- **Entity Attribute Binding**: Link sentiments to specific aspects (e.g., "Battery → Great", "Screen → Bad") to avoid false positives where a user praises one feature but critiques another

### Branch B: Psychological Manipulation Signals

- **Urgency Quantifiers**: Regex extraction for scarcity patterns (e.g., "only \d+ left", "deadline", "expires soon")
- **Social Proof Exploitation**: Detection of unsubstantiated consensus phrases ("everyone is buying", "best seller", "sold out everywhere") intended to trigger FOMO (Fear Of Missing Out)
- **Superlative Density**: Ratio of extreme adjectives ("miracle", "perfect", "life changing", "best ever") per sentence, which correlates with paid/inauthentic reviews
- **Emotional Volatility**: Sentiment swings within a single review (extremely positive → fearful) which often indicates manipulation tactics designed to create urgency through emotional manipulation

### Branch C: Stylistic Fingerprinting Signals

- **Function Word Distribution**: Frequency vectors of "invisible" words (articles, prepositions, pronouns like the, and, of) which represent unconscious cognitive habits difficult to disguise
- **Syntactic N-Grams**: Sequences of Parts of Speech tags (e.g., Adjective → Noun → Verb) to model sentence structure independent of topic
- **Punctuation & Formatting Habits**: Specific ratios of exclamation marks, usage of ellipses (...), capitalization patterns, and double spacing, which serve as a "digital handwriting" signature
- **Rare Word Reuse**: Track uncommon vocabulary (e.g., "plethora", "egregious") across allegedly different users—natural users rarely share rare words by chance

---

## 4. Model Selection Rationale

### Branch A: Contradiction Detection → Cross Encoder (RoBERTa Large MNLI)

**Challenge Addressed**: "Battery lasts forever" vs "Charge twice daily"

**Why**: We require deep logical understanding, not just topic similarity. Standard Bi Encoders (using Cosine Similarity) are faster but fail at logic—distinguishing "good" from "not good" is difficult because they focus on topic similarity rather than logical relationships.

**Rationale**: A Cross Encoder processes both sentences simultaneously, allowing the self attention mechanism to "see" the logical relationship (Entailment vs. Contradiction) directly. We use RoBERTa-Large-MNLI pre trained on MultiNLI (433k sentence pairs with entailment/contradiction/neutral labels), making it the state of the art for logical reasoning tasks.

**Optimization Note**: For production latency concerns, we can use a bi encoder for fast candidate retrieval, then apply the cross encoder only to high potential contradiction pairs.

### Branch B: Manipulation Detection → DistilBERT (Fine tuned)

**Challenge Addressed**: "Only 2 left! Buy now!"

**Why**: Psychological triggers rely on subtle context and phrasing (e.g., "You should too before they're gone!") which simple keyword lists might miss or flag falsely.

**Rationale**: DistilBERT provides the contextual understanding of a Transformer model but is 40% smaller and 60% faster than BERT Base, ensuring we can detect sophisticated manipulation patterns with low latency.

**Zero Shot Fallback**: we can use LLM prompting to bootstrap training labels if supervised data is unavailable.

### Branch C: Stylistic Fingerprinting → Unsupervised Distance Metric (Burrows' Delta)

**Challenge Addressed**: Multiple "different" users sharing one authorial voice

**Why**: We cannot use a standard classifier because we don't have labeled classes for every potential fraudster. We cannot train a supervised model asking "Is this User X?" because we don't have historical data for every potential bad actor.

**Rationale**: Instead, we map each review to a "Style Vector" in high dimensional space. If the Euclidean distance (or Burrows' Delta metric) between the Style Vectors of User A and User B is below a threshold—meaning they are statistically indistinguishable—we flag them as a sockpuppet cluster.

**Threshold Calibration**: We compute the distribution of Delta distances for known legitimate multi review users and set the threshold at the 5th percentile, capturing statistical outliers likely to be sockpuppets while minimizing false positives.

**Scalability Note**: For larger datasets, we can use dimensionality reduction (PCA on style vectors) followed by DBSCAN clustering to automatically identify sockpuppet groups without exhaustive pairwise comparisons.

---

## 5. Aggregation & Decision Logic

The **Risk Aggregation Engine** combines the signals from all three branches:

### Weighted Scoring

- **Contradiction (Branch A)**: Weight 0.5 (high certainty indicator—logical contradictions are objective)
- **Manipulation (Branch B)**: Weight 0.3 (behavioral indicator context dependent)
- **Fingerprinting (Branch C)**: Weight 0.2 (metadata indicator requires multiple reviews to confirm)

### Veto Logic

If any single branch outputs a confidence score > 0.95 (e.g., blatant self contradiction like "never breaks" + "broke immediately"), the review is flagged regardless of the other branches.

### Interpretability

The output JSON includes the `trigger_source` field (e.g., "Flagged by Branch A: Contradiction found in sentences 1 and 3") to aid human review and build user trust in the system.


### Confidence Calibration

We apply **temperature scaling** post training to ensure output probabilities are properly calibrated—a 0.95 confidence score should indicate 95% empirical accuracy.

---

## 6. Production Considerations

### Latency

- **Branch A** is the most expensive (O(N²) complexity for sentence pairs). We optimize this later using filtering heuristics (only check pairs with semantic overlap)
- **Target**: < 50ms per review

### Scalability

- The three branches run **asynchronously** in parallel
- **Branch C** (Stylometry) is purely statistical and can run near instantly
- **Branch B** (Manipulation) uses lightweight DistilBERT for fast inference

### Monitoring

- Track per branch performance metrics separately to identify drift in specific attack patterns
- Monitor false positive rates by branch to tune weights over time
- Log flagged reviews for periodic human audit

### Validation Approach

To ensure pipeline effectiveness before deployment:

- **Branch A**: Test on SNLI/MultiNLI dev sets + synthetic contradictions from the provided dataset
- **Branch B**: Create 100 human annotated manipulation examples for validation
- **Branch C**: Use known sockpuppet cases from public datasets (e.g., Ott et al. 2011 deceptive opinion spam corpus)

**Success Metrics**:
- Precision > 0.85 (minimize false positives to maintain user trust)
- Recall > 0.70 (catch majority of deceptive reviews)
- Latency < 200ms per review initially, optimized to < 50ms

---

## References

- **MultiNLI Dataset**: Williams et al. (2018) - Natural Language Inference corpus
- **Burrows' Delta**: Burrows (2002) - Delta: A measure of stylistic difference
- **Deceptive Review Detection**: Ott et al. (2011) - Finding deceptive opinion spam
- **DistilBERT**: Sanh et al. (2019) - Smaller, faster, cheaper transformer models
