# Part 3: Zero Shot Manipulation Detection
---

## Approach 1: Weak Supervision with Snorkel + SetFit Hybrid

### 1. The Technique

We combine **programmatic weak supervision** (Snorkel framework) with **few-shot contrastive learning** (SetFit) to create a self improving manipulation detector that requires zero manual labels.

**Stage 1 - Weak Supervision (Snorkel):**

We write 15-20 **Labeling Functions (LFs)** - Python functions that programmatically vote on whether a review is manipulative:

```python
@labeling_function()
def lf_scarcity(review):
    """Detect false scarcity patterns"""
    if re.search(r"only \d+ (left|remaining)|selling out fast", review.text.lower()):
        return MANIPULATIVE
    return ABSTAIN

@labeling_function()
def lf_urgency_temporal(review):
    """Detect time-pressure tactics"""
    urgency_words = ["expires", "deadline", "today only", "limited time", "hurry"]
    if any(word in review.text.lower() for word in urgency_words):
        return MANIPULATIVE
    return ABSTAIN

@labeling_function()
def lf_sentiment_volatility(review):
    """Detect emotional manipulation via sentiment swings"""
    sentences = sent_tokenize(review.text)
    sentiments = [vader.polarity_scores(s)['compound'] for s in sentences]
    if len(sentiments) >= 3 and np.std(sentiments) > 0.6:
        return MANIPULATIVE
    return ABSTAIN

@labeling_function()
def lf_caps_excess(review):
    """Detect shouting (excessive capitalization)"""
    caps_ratio = sum(1 for c in review.text if c.isupper()) / max(len(review.text), 1)
    if caps_ratio > 0.30: 
        return MANIPULATIVE
    return ABSTAIN

@labeling_function()
def lf_exclamation_density(review):
    """Detect excessive excitement markers"""
    exclamations = review.text.count('!')
    words = len(review.text.split())
    if words > 0 and (exclamations / words) > 0.03: 
        return MANIPULATIVE
    return ABSTAIN

@labeling_function()
def lf_superlative_overload(review):
    """Detect extreme adjective spam"""
    superlatives = ['amazing', 'perfect', 'incredible', 'life-changing', 
                    'miracle', 'best ever', 'unbelievable', 'revolutionary']
    count = sum(review.text.lower().count(word) for word in superlatives)
    if count >= 3:  
        return MANIPULATIVE
    return ABSTAIN

@labeling_function()
def lf_social_proof_spam(review):
    """Detect unverifiable consensus claims"""
    patterns = ['everyone is buying', 'bestseller', '#1 rated', 'sold out everywhere',
                'thousands of satisfied', 'most popular', 'everyone loves']
    if any(pattern in review.text.lower() for pattern in patterns):
        return MANIPULATIVE
    return ABSTAIN
```

**Label Aggregation:** Snorkel's generative model analyzes agreement patterns between these LFs (without any ground truth labels) and produces probabilistic training labels for the entire dataset. For example: "Review #523: 0.87 probability manipulative (agreed by LF_SCARCITY, LF_CAPS_EXCESS, and LF_EXCLAMATION_DENSITY)".

**Stage 2 - Few Shot Fine Tuning (SetFit):**

We use Snorkel's probabilistic labels to fine tune a lightweight Sentence Transformer model (SetFit framework):

1. **Contrastive Pair Generation:** SetFit creates positive/negative pairs from weakly labeled reviews
2. **Embedding Fine tuning:** Train a small Sentence Transformer (e.g., `all-MiniLM-L6-v2`, 23MB) to separate manipulative vs. legitimate reviews in embedding space
3. **Classification Head:** Add a simple logistic regression classifier on top

**Stage 3 - Self Improvement Loop:**

When users flag false positives/negatives (thumbs down), we:
1. Add corrections to a "gold standard" validation set
2. Weekly A/B test different LF threshold configurations
3. Automatically deploy the best performing configuration
4. Retrain SetFit monthly with accumulated feedback as hard negatives

---

### 2. Pros & Cons

**Pros:**
-  **Zero API costs** - Snorkel (open source Python) + SetFit (local HuggingFace models) = $0
-  **Generates 10,000+ training labels automatically** - From just 15-20 LFs in <1 hour of development
-  **Highly interpretable** - Output includes: "Flagged by LF_SCARCITY (confidence: 0.92) and LF_CAPS_EXCESS (confidence: 0.78)"
-  **Self improving via user feedback** - Corrections automatically refine LF weights through Snorkel's statistical modeling
-  **Production grade speed** - SetFit inference: ~15ms per review (vs. 100ms for NLI cross encoders)
-  **Robust to domain shift** - Add category specific LFs in minutes (e.g., `lf_fake_sizing` for fashion reviews)
-  **State of the art academic backing** - Snorkel has been deployed at Google, Intel, DARPA, and Stanford Medicine
-  **No expertise required** - Writing LFs is intuitive (just Python + regex)

**Cons:**
-  **Initial LF development time** - Requires 2-3 days to design and test quality labeling functions
-  **Threshold tuning needed** - Requires small validation set (~100 reviews) to calibrate LF confidence scores
-  **Misses subtle manipulation** - Purely syntactic LFs won't catch sophisticated psychological tactics that lack explicit linguistic markers (e.g., "I'd secure one soon" = indirect urgency)

---

### 3. Expected Performance (with Reasoning)

**Week 1 (15 LFs + Snorkel aggregation):**
- **Precision: 0.80-0.85** - Conservative LFs produce few false positives
- **Recall: 0.60-0.65** - Misses subtle manipulation without explicit markers
- **F1 Score: 0.70-0.73**

**Week 2 (+ SetFit distillation):**
- **Precision: 0.78-0.83** - Model generalizes beyond LF patterns (slight precision drop but catches edge cases)
- **Recall: 0.72-0.78** - Model learns implicit patterns not captured by rules
- **F1 Score: 0.75-0.80** (+7% improvement)

**Post-Deployment (Month 1+ with user feedback - continuous improvement):**
- **Precision: 0.82-0.87**
- **Recall: 0.76-0.82**
- **F1 Score: 0.80-0.84** (+5% improvement from continuous learning)

**Reasoning:**
- Academic studies show Snorkel achieves **132% average improvements** over heuristic only baselines (Ratner et al. 2017)
- SetFit with pseudo labeled data typically achieves **85-90%** of fully supervised model performance (Tunstall et al. 2022)
- Our hybrid approach: Snorkel generates ~10k "pseudo examples" → SetFit treats as real training data → Expected **85% of supervised baseline ≈ F1 0.80**

**Performance by Manipulation Type:**

| Manipulation Type | Snorkel-Only F1 | Snorkel + SetFit F1 | Explanation |
|---|---|---|---|
| Obvious scarcity ("Only 2 left!") | 0.90 | 0.92 | Explicit LF captures it perfectly |
| Caps lock spam ("AMAZING!!!") | 0.85 | 0.88 | Statistical LF + model reinforcement |
| Subtle urgency ("Consider soon") | 0.55 | 0.75 | Model learns implicit urgency patterns |
| Emotional volatility (sentiment swings) | 0.70 | 0.82 | Sentiment LF + contextual embeddings |

---

### 4. Validation Strategy

**Phase 1 - Snorkel LF Quality Check (Day 3):**

1. **Synthetic Positive Set Test:**
   - Manually create 50 obvious manipulation examples (e.g., "ONLY 1 LEFT!!! BUY NOW OR REGRET FOREVER!!!")
   - Target: **LF Coverage >80%** (LFs should flag 40+ of these reviews)
   - Method: Check which LFs trigger for each review, identify gaps

2. **Synthetic Negative Set Test:**
   - Use 50 known legitimate reviews from Part 2's dataset
   - Target: **False Positive Rate <15%** (flag fewer than 8 reviews)
   - Method: Ensure conservative thresholds don't over trigger

**Phase 2 - SetFit Generalization Test (Week 2):**

3. **Hold out Test Set:**
   - Reserve 200 reviews (100 manipulative, 100 legitimate) hand labeled by 2 independent annotators
   - Ensure inter annotator reliability: Cohen's Kappa >0.75
   - Use as final evaluation benchmark

4. **Ablation Study:** Compare three approaches on the hold-out set:
   - **Snorkel only:** Majority vote of LFs
   - **SetFit only:** Trained on Snorkel's probabilistic labels
   - **Hybrid ensemble:** Weighted average (0.3 × Snorkel + 0.7 × SetFit)
   - **Expected:** Hybrid beats both individual approaches by 5-8% F1

**Phase 3 - Production A/B Test (Week 3):**

5. **Deploy to 10% of Live Traffic:**
   - **Metric 1:** User engagement - Track "Was this helpful?" thumbs up/down on flagged reviews
   - **Metric 2:** Business impact - Compare purchase conversion rates on flagged vs. non flagged products
   - **Hypothesis:** Flagged reviews receive **25% more "helpful" votes** (users appreciate fraud warnings)

**Phase 4 - Adversarial Red Team Testing (Week 4):**

6. **Craft 30 Evasion Attempts:**
   - **Subtle scarcity:** "I'd recommend checking stock often" (no explicit "only X left")
   - **Disguised caps:** "A-M-A-Z-I-N-G" (character spacing to evade caps ratio detection)
   - **Implied urgency:** "You might want to act soon" (indirect suggestion)
   
7. **Measure Evasion Success Rate:**
   - Target: <20% evade detection
   - For each evasion, iterate by adding targeted LF (e.g., `lf_subtle_urgency` with semantic patterns like "recommend soon" + future tense verbs)

**Phase 5 - Long term Self Improvement Validation (Post Deployment - Month 1+):**

8. **User Feedback Integration:**
   - Collect 500+ user corrections (thumbs down on false positives/negatives)
   - Retrain SetFit monthly with corrections as "hard negatives"
   - Plot F1 trajectory over time
   - **Expected:** +2-3% F1 improvement per month for first 3 months after deployment, then plateau at 0.83-0.85

9. **LF Performance Monitoring:**
   - Track per-LF precision/recall weekly
   - Automatically disable LFs that drop below 0.60 precision
   - A/B test new LF candidates against production baseline

---

## Approach 2: Zero Shot NLI with Multi Hypothesis Inference

### 1. The Technique

We repurpose the **existing RoBERTa Large MNLI model from Part 2** as a zero shot manipulation classifier by framing detection as a Natural Language Inference (NLI) problem with multiple targeted hypotheses.

**Step 1 - Multi Hypothesis Template Library:**

Instead of a single generic hypothesis, we create a **battery of 10 specialized hypotheses** targeting distinct manipulation tactics:

```python
MANIPULATION_HYPOTHESES = [
    "This review creates false urgency or artificial scarcity.",
    "This review pressures readers to act immediately without justification.",
    
    "This review uses extreme emotional language to manipulate readers.",
    "This review exploits fear or FOMO (fear of missing out).",
    
    "This review makes unverifiable claims about popularity or consensus.",
    "This review falsely claims widespread adoption or bestseller status.",
    
    "This review uses excessive superlatives without substantiation.",
    "This review contains impossibly perfect claims that seem fabricated.",
    
    "This review tries too hard to sound genuine and authentic.",
    "This review contains markers of paid or incentivized content."
]
```

**Step 2 - Multi Hypothesis NLI Inference:**

For each review, we run NLI inference against all 10 hypotheses in parallel:

```python
entailment_scores = []

for hypothesis in MANIPULATION_HYPOTHESES:
    # Run NLI cross encoder (same model)
    scores = nli_model.predict([(review_text, hypothesis)])
    
    # Extract entailment probability (scores[0] = [contradiction, entailment, neutral])
    entailment_prob = softmax(scores[0])[1]  
    entailment_scores.append(entailment_prob)
```

**Step 3 - Aggregation via Max Voting:**

```python
# If ANY hypothesis shows strong entailment, flag as manipulation
max_entailment = max(entailment_scores)
is_manipulative = max_entailment > 0.75  

confidence = max_entailment

# which hypothesis triggered
trigger_idx = np.argmax(entailment_scores)
explanation = f"Flagged: entailment with '{MANIPULATION_HYPOTHESES[trigger_idx]}' (confidence: {max_entailment:.2f})"
```

**Step 4 - Optional Distillation for Speed (Week 2):**

Once validated, we can distill the slow NLI cross encoder into a fast **DistilBERT classifier**:

1. Use NLI predictions as pseudo labels for 10,000 unlabeled reviews
2. Train DistilBERT classifier on these labels
3. **Result:** 6-7x faster inference (~15ms vs. 100ms), ~95% accuracy retention

---

### 2. Pros & Cons

**Pros:**
-  **Zero training required** - Uses Part 2's existing NLI model directly (no additional setup)
-  **No API costs** - Local RoBERTa Large MNLI inference only
-  **Semantic understanding** - Catches subtle manipulation through logical reasoning (e.g., "I'd secure one soon" detected as indirect urgency)
-  **Highly interpretable** - "Flagged because: entailment with hypothesis 'creates false urgency' (confidence: 0.87)"
-  **Flexible & extensible** - Add new manipulation types by writing new hypotheses (no retraining needed)
-  **Improves with user feedback** - User corrections refine hypothesis library + can retrain distilled model
-  **Fast deployment** - Can go live in 2-3 days (hypothesis design + threshold tuning)
-  **Proven academic approach** - Zero shot NLI classification achieves 85-92% accuracy on unseen labels

**Cons:**
-  **Initial inference latency** - Cross encoder is slow (~100ms per review × 10 hypotheses = 1 second)
  - **Mitigation Option 1:** Parallel GPU processing → 100ms total
  - **Mitigation Option 2:** Distill to DistilBERT after validation → 15ms
-  **Hypothesis design sensitivity** - Poorly worded hypotheses lead to false positives/negatives
  - **Mitigation:** A/B test 5 paraphrase variations of each hypothesis on small validation set, pick best performers
-  **May miss statistical patterns** - NLI struggles with non semantic manipulation (e.g., excessive punctuation "!!!!!!")
  - **Mitigation:** Hybrid approach with simple regex post filters (e.g., flag if >5 exclamation marks regardless of NLI score)

---

### 3. Expected Performance (with Reasoning)

**Week 1 (Zero-shot NLI with 10 hypotheses):**
- **Precision: 0.75-0.82** - NLI is conservative, produces fewer false positives than keyword rules
- **Recall: 0.68-0.75** - Catches semantic manipulation but misses purely stylistic signals
- **F1 Score: 0.72-0.78**

**Week 2 (+ Hypothesis optimization via A/B testing):**
- Test 50 hypothesis paraphrase variations on 200 sample validation set
- Select best 10 performers based on individual F1 scores
- **F1 Score: 0.76-0.82** (+5% improvement)

**Post Deployment (Week 3+ with distillation for speed):**
- Train DistilBERT on 10,000 NLI labeled reviews
- Model learns to generalize beyond explicit hypothesis patterns
- **Precision: 0.78-0.84**
- **Recall: 0.74-0.80**
- **F1 Score: 0.76-0.82** (comparable performance, 6-7x faster)

**Reasoning:**
- Zero shot NLI transformers achieve **85-95% accuracy** out of box on classification tasks (Yin et al. 2019)
- Multi hypothesis approach adds robustness: if one hypothesis fails, 9 others compensate → Expected +5-10% over single hypothesis baseline
- Distillation typically retains **90-95%** of teacher model performance (Hinton et al. 2015)
- **Combined estimate:** 0.85 (zero shot baseline) × 1.05 (multi hypothesis boost) × 0.95 (distillation retention) ≈ **F1 0.78-0.82**

**Performance by Manipulation Type:**

| Manipulation Type | Single Hypothesis F1 | Multi-Hypothesis F1 | Explanation |
|---|---|---|---|
| Explicit urgency ("Act now!") | 0.82 | 0.88 | Clear semantic match with urgency hypothesis |
| Subtle FOMO ("You might miss out") | 0.68 | 0.80 | Multiple hypotheses capture nuanced implication |
| Unverifiable social proof | 0.75 | 0.84 | Dedicated "social proof spam" hypothesis specialized for this |
| Excessive superlatives ("BEST EVER") | 0.65 | 0.76 | "Exaggeration" hypothesis + caps detection |

---

### 4. Validation Strategy

**Phase 1 - Hypothesis Quality Assessment (Day 2):**

1. **Single Hypothesis Baseline Test:**
   - Test each of the 10 hypotheses individually on 100 known manipulation examples
   - Rank hypotheses by precision and recall
   - **Action:** Drop bottom 3 performers (likely poorly worded hypotheses)

2. **Hypothesis Refinement via Paraphrasing:**
   - For each low performing hypothesis (F1 <0.60), generate 5 paraphrase alternatives using GPT-4:
   
   ```
   Original (F1=0.58): "This review uses emotional manipulation."
   
   Alternative 1: "This review exploits readers' emotions to influence their purchasing decision."
   Alternative 2: "This review triggers strong feelings (fear, excitement, urgency) without factual basis."
   Alternative 3: "This review manipulates emotional responses rather than providing objective information."
   ```
   
   - Test all 5 alternatives on validation set, select best performer

**Phase 2 - Threshold Calibration (Week 1):**

3. **ROC Curve Analysis:**
   - On 200 sample validation set, test entailment thresholds: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
   - Plot precision recall curve
   - **Goal:** Find optimal threshold that maximizes F1 (likely ~0.75)

4. **Per-Hypothesis Threshold Tuning:**
   - Some hypotheses may need different sensitivity levels:
     - "False urgency detection": threshold 0.70 (more sensitive - catch subtle urgency)
     - "Unverifiable claims": threshold 0.80 (more conservative - avoid flagging subjective opinions)
   - Fine tune each hypothesis threshold independently

**Phase 3 - Distillation Quality Check (Week 2):**

5. **Teacher Student Agreement Test:**
   - Compare NLI cross encoder (teacher) vs. DistilBERT (student) predictions on 1,000 held-out reviews
   - **Target:** >95% label agreement
   - **Method:** Cohen's Kappa coefficient

6. **Latency Benchmark:**
   - Measure inference time on production hardware:
     - NLI cross encoder (10 hypotheses parallel): ~100ms
     - DistilBERT distilled model: ~15ms
   - **Expected:** 6-7x speedup with <5% accuracy loss

**Phase 4 - Live A/B Testing (Week 3):**

7. **Deploy Both Variants to 5% Traffic Each:**
   - **Variant A:** NLI multi hypothesis (slower, potentially more accurate)
   - **Variant B:** Distilled DistilBERT (faster, slightly less accurate)
   
8. **Compare Metrics:**
   - User satisfaction: "Was this helpful?" votes on flagged reviews
   - System latency: End to end processing time (target: <50ms)
   - **Decision Rule:** If DistilBERT F1 is within 3% of NLI and latency is <50ms, deploy DistilBERT permanently

**Phase 5 - Adversarial Robustness Testing (Week 4):**

9. **Red Team Evasion Attempts (50 samples):**
   - **Grammatical evasion:** "Consider securing your order soon" (grammatically correct but manipulative)
   - **Implication based manipulation:** "I wouldn't wait if I were you" (indirect urgency)
   - **Subtle social proof:** "My friends all grabbed one" (unverifiable but not explicitly claiming consensus)

10. **Hypothesis Expansion:**
    - For each successful evasion, design targeted hypothesis:
      - Add: "This review implies urgency through indirect suggestion or advice."
      - Add: "This review uses personal anecdotes to create false social proof."
    - Re test with expanded hypothesis library (now 12-15 hypotheses)

**Phase 6 - Cross Domain Generalization Test (Post Deployment):**

11. **Fashion Review Transfer Test:**
    - Apply model trained/validated on electronics reviews to fashion dataset
    - **Target:** If accuracy drops <10%, hypotheses are domain agnostic
    - **If accuracy drops >15%:** Add fashion specific hypotheses (e.g., "This review makes unverifiable claims about sizing or fit that seem fabricated")

---

## Comparison Matrix

| Criterion | Approach 1: Snorkel + SetFit | Approach 2: Zero-Shot NLI |
|---|---|---|
| **No API Costs** |  $0 (local only) |  $0 (local only) |
| **Generates Training Data** |  10,000+ labels automatically |  Can label unlimited for distillation |
| **Interpretability** |  Rule level explanations + confidence |  Hypothesis level reasoning |
| **User Feedback Loop** |  Self tuning LF weights + retraining |  Hypothesis refinement + distillation retraining |
| **Deployment Speed** | 2-3 days (LF writing) | 2-3 days (hypothesis design) |
| **Initial F1 (Week 1)** | 0.70-0.73 | 0.72-0.78 |
| **Optimized F1 (Week 3)** | 0.75-0.80 | 0.76-0.82 |
| **Production Latency** | 15ms (SetFit) | 15ms (distilled) / 100ms (NLI) |
| **Handles Subtle Manipulation** |  Limited (needs explicit LF) |  Good (semantic reasoning) |
| **Handles Statistical Patterns** |  Excellent (regex + stats) |  Limited (needs post filters) |
| **Domain Transfer** |  Add domain LFs in minutes |  Add domain hypotheses in minutes |
| **Best Use Case** | Statistical + explicit manipulation | Semantic + implicit manipulation |

---

## My Deployment Strategy: Hybrid Ensemble

### We can Use Both They're Complementary!

| Manipulation Type | Best Detected By |
|---|---|
| "ONLY 2 LEFT!!!" (explicit scarcity) | **Approach 1** (LF_SCARCITY) |
| "I'd secure one soon" (subtle urgency) | **Approach 2** (NLI semantic reasoning) |
| Excessive punctuation "!!!???" | **Approach 1** (LF_EXCLAMATION_DENSITY) |
| "You might regret waiting" (implied FOMO) | **Approach 2** (NLI FOMO hypothesis) |
| Sentiment volatility (emotional swings) | **Approach 1** (LF_SENTIMENT_VOLATILITY) |
| Unverifiable social proof | **Both** (LF + hypothesis reinforce) |

### Ensemble Implementation

**Days 1-3: Fast Baseline Deployment**
- Deploy **Approach 2 (NLI)** immediately (uses existing Part 2 model)
- Achieve baseline F1 ~0.75 with zero training
- **Status: Production ready**

**Days 4-10: Accuracy Optimization**
- Develop **Approach 1 (Snorkel + SetFit)** in parallel
- Write and test 15-20 LFs targeting explicit patterns NLI misses
- **Status: Ready to enhance baseline**

**Days 11-14: Production Ensemble Deployment**
```python
def ensemble_classify(review):
    # Get predictions from both models
    nli_score = nli_multi_hypothesis_model(review)
    snorkel_setfit_score = weak_supervision_model(review)
    
    # Weighted average (weights tuned on validation set)
    # NLI gets 40% weight (good at subtle cases)
    # Snorkel+SetFit gets 60% weight (good at explicit patterns + faster)
    final_score = 0.4 * nli_score + 0.6 * snorkel_setfit_score
    
    # Veto rule: If EITHER model has very high confidence, flag immediately
    # (one strong signal is enough)
    if nli_score > 0.95 or snorkel_setfit_score > 0.95:
        return True, max(nli_score, snorkel_setfit_score)
    
    # Otherwise use weighted average
    return final_score > 0.70, final_score
```

### Expected Ensemble Performance

**At 2 Week Deployment (Production Ready):**
- **Precision: 0.85-0.90** - Two models must agree → fewer false positives  
- **Recall: 0.80-0.85** - If one model misses, the other compensates  
- **F1 Score: 0.82-0.87** - **5-7% better than individual models**

**Post Deployment (Continuous Improvement):**
- **Month 1:** F1 ~0.83-0.88 (with initial user feedback)
- **Month 3:** F1 ~0.85-0.90 (mature system with accumulated learning)

**Why the Ensemble Works:**
- **Snorkel catches:** Explicit patterns (caps, punctuation, keywords)
- **NLI catches:** Implicit manipulation (subtle urgency, implied FOMO)
- **Overlap:** When both agree, confidence is very high (precision ~0.92)
- **Disagreement:** When only one flags, use higher threshold (reduce false positives)

---

## Deployment Timeline

| Week | Milestone | Expected F1 |
|---|---|---|
| **Day 1-3** | Deploy Approach 2 (NLI) - fast baseline | 0.72-0.78 |
| **Day 4-10** | Develop & test Approach 1 (Snorkel LFs) | 0.75-0.80 |
| **Day 11-14** | Deploy ensemble to production  **MEETS 2 WEEK DEADLINE** | 0.82-0.87 |
| **Post Deployment (Month 1)** | First user feedback integration (continuous improvement) | 0.83-0.88 |
| **Post Deployment (Month 3)** | Mature system with accumulated learning | 0.85-0.90 |
