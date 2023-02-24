# DFGC-VRA
Starker-kit for the DFGC-VRA competition

This code follows the workflow of [this paper](https://arxiv.org/abs/2302.00918),  you may check the paper for more details.


## Usage

### Video pre-processing
We crop the videos and keep only the facial region.

Relative coordinates of the bounding box for each video is provided in the *crop* folder, with the top left corner set as the origin.

### Feature extraction
Here we use the 1st-place solution in DFGC-2022 detection track (DFGC-1st) as the extractor.

Run the scrip bellow to get video level features:
```
python DFGC1st_feats.py
```

### Feature selection
In this step we first decide the dimension of the selected features, then perform the selection.

For dimention seletion, run:
```
python feats_num_select.py
```

For feature selection with a given dimention, run: 
```
python feats_select.py
```
