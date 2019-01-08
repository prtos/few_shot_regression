## Meta-qsar dataset

This is the processed dataset from the [meta-qsar paper](https://arxiv.org/pdf/1709.03854.pdf). The number of non-duplicated targets does not match the one described in the paper.

Original dataset can be found at: https://www.openml.org/s/13/data and can be downloaded using the [openml-python API](https://github.com/openml/openml-python)

## Format

The format of this dataset is similar to the binding-db dataset. Each file represents a single target, and the filename corresponds to `[readout]_[Target ChemBL-ID]_[Number of compounds]`

The inside of each file looks like this:
```
CHEMBL3200      Q9Y3R4  Sialidase-2
MASLPVLQKESVFQSGAHAYRIPALLYLPGQQSLLAFAEQRASKKDEHAELIVLRRGDYDAPTHQVQWQAQEVVAQARLDGHRSMNPCPLYDAQTGTLFLFFIAIPGQVTEQQQLQTRANVTRLCQVTSTDHGRTWSSPRDLTDAAIGPAYREWSTFAVGPGHCLQLHDRARSLVVPAYAYRKLHPIQRPIPSAFCFLSHDHGRTWARGHFVAQDTLECQVAEVETGEQRVVTLNARSHLRARVQAQSTNDGLDFQESQLVKKLVEPPPQGCQGSVISFPSPRSGPGSPAQWLLYTHPTHSWQRADLGAYLNPRPPAPEAWSEPVLLAKGSCAYSDLQSMGTGPDGSPLFGCLYEANDYEEIVFLMFTLKQAFPAEYLPQ
CHEMBL1089732   OC(=O)c1cc(O)c2C(=O)c3c(O)c(Cl)c(O)cc3C(=O)c2c1 4.0
CHEMBL1234524   CC(=O)N[C@@H]1[C@@H](O)C[C@@](Oc2ccc3C(=CC(=O)Oc3c2)C)(O[C@H]1[C@H](O)[C@H](O)CO)C(=O)O 2.7460000515
.
.
.
```

Where the first line corresponds to the target ChemBL ID, the corresponding accession number in uniprot and the protein name.
The second line is the target sequence, if available. In a few case, the target is actually a protein complex, and only one member of such complex is returned. There is indeed a limitation for the choosen file format, and using an older ChemBL (v17) release to match the manuscript did not help.
The remaining lines correspond to the actual dataset, with each compound chemID, it's smile (canonical) and the activity value.

__Note that I used the ChemBL release 17 to match the original paper, but some chem ID might have changed in later version._
 