# Cross-lingual Natural Language Inference with XLM-RoBERTa
This project studies how well a multilingual transformer generalizes logical reasoning across languages.
We fine-tune **XLM-RoBERTa-base** on **English** Natural Language Inference (NLI) and evaluate **zero-shot transfer** to **Hindi** using the **XNLI benchmark**.

The goal is to quantify and analyze the **cross-lingual transfer gap** in multilingual models.

Repo used: **https://github.com/microsoft/Multilingual-Model-Transfer**

---

## Dataset
**XNLI (Cross-lingual Natural Language Inference)**  
We use **XNLI**, a standard benchmark for multilingual NLI.
Each sample consists of a **premise–hypothesis pair** labeled as:

1. Entailment
2. Neutral
3. Contradiction

Languages used in this study:

**English** (training + in-language evaluation)
**Hindi** (zero-shot cross-lingual evaluation)

---

## Model
- **xlm-roberta-base**
- Transformer-based sequence classification head  
- 3-way classification: *entailment / neutral / contradiction*

---

## Methodology
1. Fine-tune XLM-RoBERTa on English XNLI
2. Evaluate the trained model on:
   - English validation set (in-language performance)
   - Hindi validation set (cross-lingual transfer)
3. Compare performance degradation across languages
4. Analyze misclassifications using confusion matrices

---

## Results
**English -> English (in language)**
   Accuracy: 0.69
   Loss: 0.91

Confusion Matrix (rows = true, columns = predicted)
[[108  43  19]
 [ 13 132  20]
 [  9  51 105]]
- The strong diagonal indicates that the model learned meaningful NLI decision boundaries in English.

**English -> Hindi (in language)**
   Accuracy: 0.564
   Loss: 1.28

Confusion Matrix
[[304 443  83]
 [ 51 676 103]
 [ 35 370 425]]

---

## Cross-Lingual Transfer Analysis
- When transferred from English to Hindi, performance drops from **0.69 → 0.56**, an absolute decrease of **~13 percentage points**.

The confusion matrices reveal that:

1. The model increasingly predicts the **Neutral** class in Hindi
2. **Entailment and Contradiction** are frequently misclassified as Neutral
3. Logical inference degrades more than semantic similarity under language shift

This demonstrates that multilingual models retain partial alignment but struggle with cross-lingual reasoning.

---

## Future Work
- Training performed on CPU only
- Subset of XNLI used due to hardware constraints
- No target-language fine-tuning (pure zero-shot transfer)

---

## Conclusion
This project shows that **XLM-RoBERTa generalizes imperfectly across languages**, with a measurable and systematic *cross-lingual transfer gap* between English and Hindi.
The results highlight the need for language-adaptive fine-tuning and alignment methods in multilingual reasoning tasks.
