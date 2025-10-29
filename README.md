# Shipping-signals-with-FSL
Code for Identification of shipping signals with few-shot learning: A distribution-aware approach submitted to PLOS ONE

README — Supporting Information Package

This package provides supporting information for the PLOS ONE manuscript.
It includes two representative raw acoustic samples, code, and a data description document.
Due to file size limitations, the preprocessed spectrogram (.npy) data are provided separately as multiple compressed files uploaded individually to the submission system.

Note: Only Raw_sound_sample is included inside this ZIP.
The preprocessed spectrogram (.npy) files are uploaded as separate compressed archives.

2) Separate Attachments (submitted individually)

Submission type:
Supporting Information – Compressed/ZIP File Archive

Contents: Preprocessed spectrogram .npy data (split into multiple ZIP files)

Usage instructions:

Download all spectrogram ZIP archives and extract them into the same folder.
Recommended location:

Supporting_Information/Preprocessed_Data/Spectrogram/


or under the path defined as data_root in train_siamese.py (default /root/data/).

If using a different location, update data_root in train_siamese.py.

3) Data Summary

Location/Period: Southern sea of Jeju Island, July 15–26, 2016

Sensor/Settings: Self-recording hydrophone, 24 kHz sampling (raw)

Spectrogram preprocessing:

1 s Hamming window, 50% overlap

STFT, 1 Hz frequency resolution

Frequency range: 0–1000 Hz

Segmented into 1-second units

Converted to (224×224×3) image format, scaled 0–255, stored as .npy

Data type: Acoustic data from 21 ships (cargo, tanker, container vessels)

Included in this submission:

ZIP: 2 raw acoustic samples + code + description document

Separate attachments: preprocessed spectrogram .npy data (multiple ZIP files)

Complete raw/preprocessed dataset: Not included due to size limits.
Available upon reasonable request to the corresponding author.

4) Environment

Python 3.9–3.10 recommended

Required packages:

tensorflow >= 2.10

numpy, pandas

Example installation:

pip install "tensorflow>=2.10" numpy pandas


GPU with CUDA/CuDNN is strongly recommended.

5) Code Overview
fewshot_utils.py

Pair/Distance/Loss:
make_contrastive_pairs_tf, cosine_distance_tf, contrastive_loss

Feature/Logit processing:
calculate_dist, calculate_loss, calculate_acc

Data handling:
load_npy_files_from_folder (split train/validation, normalized 0–1),
get_samples (support/query sets per episode)

Model:
build_vgg16 (512-D embeddings),
get_siamese_model (support/query branches)

train_siamese.py

Runs 50 folds with reproducible seeds

Workflow per fold: reset model → load data → train → validate → log/save results

Saves mean validation accuracy into save_result.csv

6) How to Run

Download all separate spectrogram ZIP archives, extract into
Preprocessed_Data/Spectrogram/ or data_root path.

Adjust data_root in train_siamese.py if needed.

Run training:

python train_siamese.py


Outputs:

Training logs per epoch

save_result.csv with mean validation accuracy across folds

7) Data Availability

This submission provides:

Two raw acoustic samples (inside this ZIP)

Preprocessed spectrogram data (.npy) as separate uploaded archives

The complete dataset is available from the corresponding author upon reasonable request.
