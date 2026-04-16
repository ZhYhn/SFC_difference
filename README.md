# Structure-Function_Coupling_VGCL

This is the code for the bioRxiv preprint: ***[Informational and methodological differences in regional structure-function coupling in modeling approaches](https://doi.org/10.1101/2025.10.26.684597)***.

including: 
- Code for calculating structural connectivity matrix and functional connectivity matrix
- Code for all models used in the paper
- Code for calculating differences
- Code for visualization

The data for the paper can be found here: https://doi.org/10.5281/zenodo.17828069

## Dependencies

The core environment for implementing graph neural network is:
- **Python** 3.12.11
- **PyTorch** 2.8.0+cu129
- **PyTorch Geometric (PyG)** 2.6.1

## `01_Preprocessing`
- `compute_fc.py`: Calculate the average time series within the brain region and compute the Pearson correlation coefficient.
- `fslr_to_fsaverage.sh`: Convert the rs-fMRI data from the fs_LR space to the fsaverage space (refer to [HCP Users FAQ](https://wiki.humanconnectome.org/docs/HCP%20Users%20FAQ.html) item 9: *How do I map data between FreeSurfer and HCP?*).
- `generateFC.sh`: Run the scripts `compute_fc.py` and `fslr_to_fsaverage.sh` to generate functional connectivity matrices.
- `generateSC.py`: Convert .mat files to numpy arrays to generate structural connectivity matrices (The structural connectivity matrix in .mat format come from the dataset: https://doi.org/10.5281/zenodo.3928848).

## `02_Train`
- `full_direct_train.py`: Train the full models and direct models, including: MLP, pGCN, and sGCN.

## `03_Predict`
- `full_direct_predict.py`: Use the previously obtained models for prediction, the predictive models (regression, MLP, pGCN) yield the predicted connectivity, and sGCN directly yields SFC.
- `get_sfc_from_predConn.py`: Obtain the SFC calculated by all approaches.

## `04_Difference`
- `info_meth_ttest.py`: At SFC level, t-tests were performed to determine the proportion of cortical regions where the informational difference was significantly greater or less than the methodological difference.
- `difference.py`: Calculate the informational difference and the methodological difference at the predicted-connectivity level.

## `05_Identification`
- `nclass_individual_identification.py`: Use the SFC calculated by all approaches for individual identification.

## `Visualization`
Draw the figures shown in the Results section of the paper.

## `utils`
Some modules used in the analysis.
