# Downloading RSNA Pneumonia Dataset

The dataset is on Kaggle. You need a Kaggle account and the Kaggle CLI.

```bash
pip install kaggle
# Place your kaggle.json API key at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

cd data/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip
```

Files you will use:
- stage_2_train_images/          (DICOM files, one per patient)
- stage_2_train_labels.csv       (patientId, x, y, width, height, Target)
- stage_2_detailed_class_info.csv (patientId, class)

The dataset has:
- 26,684 unique patient IDs
- Target=1 → Lung Opacity (pneumonia), Target=0 → No Lung Opacity

Label mapping used in eval_rsna.py:
- Any box with Target=1 for a patient → label "pneumonia"
- All boxes Target=0 for a patient → label "normal lung"
