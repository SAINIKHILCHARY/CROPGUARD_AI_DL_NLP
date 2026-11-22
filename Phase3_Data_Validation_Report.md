# CropGuard AI – Phase 3 Data Collection & Validation Report

## 1. Dataset Audit & Cleaning Summary
| Clean_Dataset       |   Classes |   Raw_Images |   Clean_Images |   Train_Imgs |   Valid_Imgs |   Test_Imgs |   Total_Boxes |   Exact_Dups |   Cross_Split_Leaks | Issues_Resolved                                                                      |
|:--------------------|----------:|-------------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|--------------------:|:-------------------------------------------------------------------------------------|
| CottonDisease_Clean |         5 |          250 |            243 |          169 |           49 |          25 |           269 |            0 |                   0 | 7 exact dup groups (5 cross-split leaks); 7 near-dup pairs; broken data.yaml path    |
| MultiCrop_Clean     |        30 |        11053 |           2191 |         1532 |          438 |         221 |          4305 |            0 |                   0 | 8,862 train images missing labels; missing valid/ folder; 2,191 valid test labels    |
| PlantDoc_Clean      |        29 |         2346 |           2332 |         1633 |          466 |         233 |          8419 |            0 |                   0 | No val/test splits; 4 class-conflicting dup pairs; 10 unannotated leaf files         |
| RiceDisease_Clean   |         3 |          850 |            699 |          492 |          137 |          70 |          2690 |            0 |                   0 | val->test YAML defect; 111 cross-split exact dups; 17 Bacterial Blight box conflicts |

## 2. Validation Checks Passed Across All Clean Datasets
- **File Pairing:** 100% 1:1 image-to-YOLO `.txt` label alignment across `train`, `valid`, and `test`.
- **Annotation Integrity:** 0 corrupt images, 0 empty label files, 0 out-of-range class IDs or normalized coordinates.
- **Zero Data Leakage:** 0 exact SHA-256 duplicates and 0 dihedral perceptual-hash near-duplicates across splits.
- **Configuration:** Every clean dataset includes a verified `data.yaml` with explicit `train`, `val`, and `test` paths.