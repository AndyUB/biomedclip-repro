# Downloading PCam Test Set

Download the test split HDF5 files from the official PCam repo:
https://github.com/basveeling/pcam

You need these two files (gzipped HDF5):
- camelyonpatch_level_2_split_test_x.h5.gz
- camelyonpatch_level_2_split_test_y.h5.gz

They are hosted on Zenodo. Direct links (from the GitHub README):

```bash
cd data/pcam

# Images (test)
wget -O camelyonpatch_level_2_split_test_x.h5.gz \
  "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz"

# Labels (test)
wget -O camelyonpatch_level_2_split_test_y.h5.gz \
  "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz"

gunzip camelyonpatch_level_2_split_test_x.h5.gz
gunzip camelyonpatch_level_2_split_test_y.h5.gz
```

After extraction you should have:
- data/pcam/camelyonpatch_level_2_split_test_x.h5
- data/pcam/camelyonpatch_level_2_split_test_y.h5

The test set has 32,768 images (96×96 RGB), balanced 50/50.
