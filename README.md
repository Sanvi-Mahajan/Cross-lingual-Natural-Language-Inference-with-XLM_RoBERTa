# Cross-lingual Natural Language Inference with XLM-RoBERTa

## Problem
Multilingual NLP models often perform well in high-resource languages such as English but degrade significantly when transferred to low-resource or code-mixed languages.  
This project studies **cross-lingual transfer** for **Natural Language Inference (NLI)**.

---

## Objective
- Fine-tune **XLM-RoBERTa** on English NLI data  
- Evaluate **zero-shot transfer** performance on Hindi  
- *(Planned)* Extend the study to **Hinglish** (code-mixed Hindi–English)

---

## Dataset
**XNLI (Cross-lingual Natural Language Inference)**  
- English: training + validation  
- Hindi: validation only (zero-shot evaluation)

---

## Model
- **xlm-roberta-base**
- Sequence classification head  
- 3-way classification: *entailment / neutral / contradiction*

---

## Methodology
1. Fine-tune XLM-RoBERTa on English XNLI
2. Evaluate the trained model on:
   - English validation set (in-language performance)
   - Hindi validation set (cross-lingual transfer)
3. Compare performance degradation across languages

---

## Preliminary Observations
- Training loss decreases steadily during fine-tuning
- Cross-lingual evaluation shows a significant performance drop, consistent with prior multilingual transfer literature

---

## Limitations
- Training performed on CPU (no GPU acceleration)
- Subset of XNLI used due to hardware and storage constraints
- Hinglish evaluation pending (dataset construction in progress)

---

## Future Work
- Construct a Hinglish NLI dataset
- Compare **EN → HI** vs **EN → HINGLISH** transfer
- Analyze error patterns by NLI label type
