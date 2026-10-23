# Applied AI - Project 2: Deep Learning Applications

## Week 6: Pneumonia Detection (Medical Imaging)

**Kaggle Notebook Link:** [[PASTE YOUR WEEK 6 KAGGLE LINK HERE](https://www.kaggle.com/code/mahumjunaid/week-6-medical-imaging-pneumonia-detection)]

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
