## Week 7: Text and Multimodal AI (Medical NLP Pipeline)

**Kaggle Notebook Link:** [[WEEK 7 KAGGLE LINK](https://www.kaggle.com/code/mahumjunaid/week-7-text-and-multimodal)]

### Overview
This week expanded my Week 6 chest X-ray model into a **multimodal medical triage system**. The goal was to prove that an AI can process a doctor's note, extract critical data (dates, dosages, hospital names), and combine it with an image prediction—all while guaranteeing patient privacy. 

### 1. Text Classification (Spam Detection)
To understand text classification, I trained models on the SMS Spam Collection dataset.
*   **Word 1-2 grams:** Macro-F1 **0.974** (Precision: 0.979, Recall: 0.933)
*   **Char 2-5 grams:** Macro-F1 **0.982**
*   **Top Features:** Spam messages relied on urgency triggers (`'call'`, `'txt'`, `'free'`, `'claim'`). Normal messages (ham) used conversational words (`'my'`, `'me'`, `'ok'`).
*   **Robustness:** The Character n-gram model successfully caught disguised spam (`"WIN a FREE priz3!"` with 97% probability) and correctly ignored Roman Urdu (`"kal milte hain"` with 12% probability).

### 2. Named Entity Recognition (NER) on Medical Data
I tested the out-of-the-box spaCy model on my own 10 medical sentences (using Pakistani hospitals like Shaukat Khanum and Aga Khan). 
*   **Base Model (F1 = 0.857):** The general model struggled. It labeled the drug **"Amoxicillin" as a location (GPE)** and **"Karachi" as an organization (ORG)**.
*   **Custom Rules (F1 = 0.667):** I added `EntityRuler` patterns for local hospitals and drugs. Interestingly, the F1 score dropped because my manual "gold standard" was initially incomplete. This was a crucial lesson in data annotation: the evaluation is only as good as your ground truth.
*   **Remaining Errors:** The model still confuses complex terms like "Aga Khan University Hospital" (missing it as a single ORG). This shows that a rule-based approach is fragile, and a domain-specific model (like BioBERT) is needed for true medical NLP.

### 3. Regex Fields & OCR
*   **Regex:** I wrote patterns to extract CNICs, mobile numbers, amounts (PKR/Rs), dates, and invoice numbers. I included **negative tests** (e.g., ensuring a 13-digit order number was not flagged as a CNIC) to prevent false positives.
*   **OCR:** Using `easyocr` on a synthetic invoice, the raw Character Error Rate (CER) was **0.032**. I wrote context rules (e.g., changing the letter 'O' to the number '0' inside numbers) which successfully reduced the CER for specific test cases to **0.000**. 

### 4. Multimodal Case Record (The Final Pipeline)
I combined my Week 6 image model and the Week 7 text pipeline into a single JSON record. 
*   **Privacy First:** Before extracting any entities, the system masks CNICs and phone numbers. The output shows `[PHONE]` in the `note_masked` field, preventing PII leakage into logs.
*   **Transparency:** The JSON includes `reasons` for the decision (e.g., "image model: P(PNEUMONIA) = 0.99" and "note mentions: fever, breath"). 
*   **Safety:** The disclaimer states: *"Decision support only; a qualified human decides."* If the image model says the patient is healthy but the note mentions "fever" and "breath," the system still flags the case for a `REFER` review, prioritizing clinical symptoms over a single X-ray.

### Week 8 Proposal
For the final project, I will deploy this multimodal pipeline using **FastAPI**. The API will accept an image upload and a text note, run the image through the ResNet model, run the text through spaCy and Regex, and return the fused JSON record with PII masked.

# Applied AI - Project 2: Deep Learning Applications

## Week 6: Pneumonia Detection (Medical Imaging)

**Kaggle Notebook Link:** [[ WEEK 6 KAGGLE LINK ](https://www.kaggle.com/code/mahumjunaid/week-6-medical-imaging-pneumonia-detection)]

**Overview:**
This week, I moved from natural scenes to a real-world medical dataset. The goal was to build a complete, defensible computer vision pipeline that handles imbalanced data, patient-level leakage, and identifies model shortcuts using Grad-CAM.

**Dataset:**
*   **Name:** Chest X-Ray Images (Pneumonia)
*   **Source:** [Kaggle - Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
*   **Classes:** 2 (NORMAL, PNEUMONIA)
*   **Total Images:** 5,863 (Train: 5,216 | Test: 624 | Val: 16)

**Data Audit & Pipeline:**
*   **Imbalance:** The dataset is heavily imbalanced (ratio 2.89). I handled this using weighted loss (NORMAL: 1.935, PNEUMONIA: 0.674).
*   **Leakage Audit:** Found 62 exact duplicates but 0 near-duplicates between train and test.
*   **Group Split:** Used `StratifiedGroupKFold` to ensure 0 patient overlap between train and validation, preventing data leakage.

**Model & Results:**
*   **Architecture:** ResNet-18 (pretrained). Phase 1 (head only) trained 1,026 parameters. Phase 2 (fine-tuned last block).
*   **Best Epoch:** Selected by Validation Macro-F1 (0.947).
*   **Test Accuracy:** 92.5%
*   **Test Macro-F1:** 0.917
*   **Threshold Tuning:** Tuned on validation for 95% recall (threshold = 0.363). Achieved 99.2% sensitivity and 80.3% specificity on the test set. PPV at 5% realistic prevalence is 20.9%.

**Grad-CAM Finding:**
While the model correctly focused on lung fields for many NORMAL images, the error gallery revealed **shortcut learning**. For some misclassified images, the model looked intensely at the bright white "R" marker or text in the corner, rather than the lungs. The proposed fix is to crop or mask these text regions before training.

### Responsible Domain AI
*   **Who could be harmed, and how?** A child with pneumonia could be sent home without treatment because of a false negative. Alternatively, a healthy patient could be unnecessarily stressed and exposed to radiation if falsely flagged (false positive).
*   **Where was it NOT validated?** The dataset consists of pediatric patients (ages 1-5) from a single hospital (Guangzhou Women and Children’s Medical Center). It has not been validated on adults, other hospitals, different X-ray machines, or the local Pakistani population.
*   **What is the human role?** The model acts as a triage assistant to prioritize a radiologist's queue. It is a screening tool to flag high-risk images; it is never a final diagnosis. A human radiologist must review every flagged image.
*   **What would change your mind?** External validation at a local Pakistani hospital demonstrating that the model maintains a high enough PPV at the local prevalence rate (accounting for local scanner differences and population demographics).

---

## Week 5: Scene Classification with Transfer Learning

**Kaggle Notebook Link:** https://www.kaggle.com/code/mahumjunaid/week-5-transfer-learning

**Overview:**
This project builds a deep learning image classifier to recognize 6 types of natural scenes (buildings, forest, glacier, mountain, sea, street) using the Intel Image Classification dataset. The model uses a pretrained ResNet-18 and explores both feature extraction and fine-tuning.

**Dataset:**
- **Name:** Intel Image Classification (puneet6060)
- **Classes:** 6 (buildings, forest, glacier, mountain, sea, street)
- **Splits:** 11,929 train / 2,105 val / 3,000 test
- **Image Size:** 224 x 224 (Resized from 150x150)

**Model Architecture:**
- **Base Model:** ResNet-18 pretrained on ImageNet
- **Phase 1 (Feature Extraction):** Froze all layers, trained only the new 6-class fully connected head. Trainable parameters: **3,078**.
- **Phase 2 (Fine-Tuning):** Unfroze `layer4` and trained with two different learning rates (1e-4 for pretrained layers, 1e-3 for the new head). Trainable parameters: **8,396,806**.

**Results:**
| Phase | Best Validation Accuracy |
| :--- | :--- |
| Feature Extraction | 91.1% |
| Fine-Tuning | 94.0% |
| **Final Test Accuracy** | **93.2%** |

**Weakest Classes (by Recall):** Mountain (0.903) and Street (0.920)
**Most Confused Pair:** Glacier vs. Mountain (61 mountain images predicted as glacier)

**Most Instructive Error:** 
In the error gallery, the model predicted "street" (confidence 0.91) for an image labeled "buildings". Looking at the image, it shows a building with a very prominent street in front of it. This is a genuinely ambiguous image where the label is technically correct, but the visual features heavily favor the "street" class.

**Next Steps: Week 6 Proposal**
For my own domain in Week 6, I propose applying this same transfer learning pipeline to a **medical imaging dataset** (specifically chest X-rays for pneumonia detection). The candidate dataset is the **"Chest X-Ray Images (Pneumonia)"** dataset from Kaggle. This will test whether ImageNet-pretrained features (edges, textures) transfer effectively to grayscale medical images, which look very different from natural scenes.
