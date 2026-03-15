# LungEvaty

## Weights
Download the weights from here: [link](https://syncandshare.lrz.de/getlink/fiHDpe7ga9znjpvRy4dFh4/)

## Installation
Requires: [uv](https://docs.astral.sh/uv/) <br>
Run the installation script: `bash install.sh`

## Data preparation
Prepare a json file with the following structure:
```json
[
    {
        "image": "data/patient_001/study_001/series_01",
        "pid": "patient_001",
        "series": "series_01",
        "y": 1,
        "time_at_event": 2,
        "y_seq": [0, 0, 1, 1, 1],
        "y_mask": [1, 1, 1, 1, 0] 
    }
]
```
Please refer to [dcm_inference.py](./dcm_inference.py) and [sample_dicom_manifest.json](./sample_dicom_manifest.json).

## Inference script
[dcm_inference.py](./dcm_inference.py) takes care of all the preprocessing from raw dicom file and can be run with the above json schema without any segmentation masks.


## Results
Outputs of the scripts will be written under `./outputs/`.