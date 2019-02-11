# Efficient molecular feature learning for low data QSAR
Official repository of the ISCB paper
## Requirements and installation
A new virtual environment with all the requirement can be created automatically by running the installation script as follows:
> bash install.sh

Otherwise, we highly recommend you to use conda for package management and make sure you met at least these requirements:
-   RDKit (version >= 2017.09)
-   Python (version >= 3.4)
-   PyTorch (version >= 0.2)

For a more complete list of requirements, please look inside the `install.sh`

## Datasets and methods

All the datasets used in our experiments are available in the `datasets` folder :
- Pubchem toxicity dataset collection at [datasets/pubchemtox/](https://github.com/prtos/few_shot_regression/blob/master/datasets/pubchemtox)
- MetaQsar dataset collection at [datasets/chembl/](https://github.com/prtos/few_shot_regression/blob/master/datasets/chembl)
- MHC dataset collection at [datasets/mhc/](https://github.com/prtos/few_shot_regression/blob/master/datasets/mhc)

All the methods are available as well :
- MetaKRR: [metalearn/models/metakrr_singlekernel.py](https://github.com/prtos/few_shot_regression/blob/master/metalearn/models/metakrr_singlekernel.py)
- RF+ECFP4 and KRR+ECFP4: [metalearn/models/fp_learner.py](https://github.com/prtos/few_shot_regression/blob/master/metalearn/models/fp_learner.py)
- Seq2Seq: [metalearn/models/seq2seq_fingerprint.py](https://github.com/prtos/few_shot_regression/blob/master/metalearn/models/seq2seq_fingerprint.py)
- IterRefLSTM: [metalearn/low_data/deepchem_meta.py](https://github.com/prtos/few_shot_regression/blob/master/metalearn/low_data/deepchem_meta.py)
- MANN: [metalearn/models/mann.py](https://github.com/prtos/few_shot_regression/blob/master/metalearn/models/mann.py)

## Experiments and training
All the experiments in the iscb paper are available in the folders `expts_iscb` and `expts_low_data`. The latter only contains the experiments related to the IterRefLSTM algorithm.
The train file in each folder should allow to train a model and save the trained weights as well as its performances on the  meta-test.

To actually train a model, please
1. Create a train.json where you specify which config list you want to use and an id indicating the config position in that list. Example of train.json
> {
>             'config_file': 'config_krr.json',
>             'config_id': 0
>    }
2. Execute the training file
> python train --config_file train.json --input_path ../datasets --output_path ./results

## Contact
[Prudencio Tossou](mailto:prudencio@invivoai.com)