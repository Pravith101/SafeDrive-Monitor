# SafeDrive Monitor model card

## Summary

SafeDrive Monitor is a research prototype that estimates three visual states from face imagery: alert, microsleep, and yawning. It includes an FL3D frame classifier and a separate UTA-RLDD video-level experiment. Neither model is validated for operational driver monitoring or safety decisions.

## Models and data

### FL3D frame classifier

- Architecture: ImageNet-pretrained MobileNetV3-Small fine-tuned on face-cropped FL3D frames.
- Labels: alert, microsleep, yawning.
- Dataset: [FL3D on Kaggle](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d), with its declared CC BY-SA 4.0 terms. The frames derive from NITYMED. See README.md for attribution and split limitations.
- Evaluation: source-video-disjoint frames; person-level separation cannot be verified from the released annotations.
- Held-out results: 90.47% accuracy, 88.51% balanced accuracy, 90.24% macro-F1. Microsleep recall is 71.37% (656 of 2,291 microsleep frames missed).

### UTA-RLDD temporal experiment

- Architecture: two-layer GRU over 30-second sequences of MediaPipe eye, mouth, and head-pose features sampled at 1 Hz.
- Labels: alert, low vigilance, drowsy. Each label describes a video's predominant state; there are no frame-level event or onset annotations.
- Data source used for the completed run: Kaggle mirrors for folds 1–4 and fold 5. The uploaders list CC0, but the mirrors are unofficial and their provenance and authority to relicense are unverified. The original [UTA-RLDD site](https://sites.google.com/view/utarldd/home) describes the dataset and requests citation; it notes that only 36 of 60 participants allowed their faces to be published. This repository contains no raw participant videos or face images.
- Evaluation protocol: five published participant-disjoint folds; threshold selection uses inner validation participants only.
- Mean held-out results across five folds: 44.7% video accuracy, 42.7% macro-F1, 35.0% drowsy-video recall, and 14.2% non-drowsy video false-warning rate. It missed 40 of 62 drowsy-labeled test videos.
- The tracked report and checkpoints are research artifacts from the completed run. They do not grant rights to the source videos or resolve the mirrors' licensing uncertainty. Obtain authorization from the dataset rights holder before reuse or redistribution where required.

## Intended use

Use for code review, education, and further research on the listed datasets. The webcam monitor is a demonstration of audio and visual state feedback only. It is not suitable for deployment in a vehicle or as a substitute for rest, stopping, or other safe-driving decisions. UTA-RLDD threshold crossings are displayed as research flags and do not produce driver warnings or sound.

## Limitations and risks

- Both datasets are small and do not establish generalization across drivers, cameras, lighting, eyewear, or real driving conditions.
- FL3D's test split separates source videos but cannot guarantee participant-disjoint evaluation.
- UTA-RLDD provides predominant video-level states, so event misses and warning delay from actual drowsiness onset cannot be measured.
- The UTA-RLDD drowsy-video recall is low. Its model must not be used to alert drivers.
- The FL3D softmax score is not a calibrated probability. The extra eye-closure and mouth cues reduce some inconsistent outputs but can miss events or fail under different conditions.
- False negatives may create unwarranted confidence; false positives may distract or alarm. Do not use the system while driving.

## Privacy and data handling

The repository does not contain raw videos or face images. Kaggle training workflows process source videos in Kaggle storage and emit feature arrays, reports, and model checkpoints. Users should not publish identifiable participant imagery, and should review the source dataset terms before sharing derived artifacts.

## Reporting

Use the exact evaluation unit and split when citing metrics. Do not compare FL3D frame-level scores directly with UTA-RLDD video-level scores or describe either result as safety validation.
