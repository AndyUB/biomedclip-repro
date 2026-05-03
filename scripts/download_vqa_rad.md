# Downloading VQA-RAD

VQA-RAD is hosted on OSF: https://osf.io/89kps/
The dataset is a single JSON file containing all 2,248 QA pairs.
Train/test split is encoded in the `phrase_type` field:
- `freeform` / `para`           → train (1,797 examples)
- `test_freeform` / `test_para` → test  (451 examples)

## Step 1 — Download the JSON

```bash
cd data/vqa_rad
curl -L "https://osf.io/download/6qdas/" -o "VQA_RAD Dataset Public.json"
```

## Step 2 — Download images via osfclient

There are 314 unique images in the OSF image folder. Use `osfclient`:

```bash
pip install osfclient
cd data/vqa_rad
osf -p 89kps clone .
# Images land at: 89kps/osfstorage/VQA_RAD Image Folder/
mv "89kps/osfstorage/VQA_RAD Image Folder" images
rm -rf 89kps
```

## Expected layout

```
data/vqa_rad/
  VQA_RAD Dataset Public.json
  images/
    synpic*.jpg   (314 images)
```
