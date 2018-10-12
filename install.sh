#!/bin/bash
#optional parameters
while [[ $# -gt 0 ]]
do
	case "$1" in
	    -n|--envname)
	    env_name="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    *)
		shift
		echo "Invalid option -$1" >&2
	    ;;
	esac
done

env_name=${env_name:metalearn}
# if conda doesn't exists install it
if ! [ -x "$(command -v conda)" ]; then
    echo "Download and install conda...."
    wget "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh
    /bin/bash miniconda.sh -b -f
    rm miniconda.sh
    echo ". ${HOME}/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    echo "conda activate base" >> ~/.bashrc
    source ~/.bashrc
fi

# create an environment and activate it
echo "create an environment for the installation and activate it"
conda create -y -n  $env_name python=3.6
#source activate $env_name
echo "conda activate $env_name" >> ~/.bashrc
# Uncomment the following lines it the last command above doesn't work
conda_location=$(type -P conda)
conda_home=$(dirname "$(dirname "$conda_location")")
export PATH=${conda_home}/envs/$env_name/bin/:$PATH

## Install some dependencies
echo "Install required dependencies in $(which python)"
pip install -U pip
pip install cython scipy numpy matplotlib
pip install scikit-learn pandas
pip install wget
pip install Keras
pip install tensorboardX
pip install pytest nose
pip install deepchem
pip install boto sagemaker
pip install ipython
pip install joblib
pip install seaborn
pip install biopython
pip install torchvision pytoune
conda install -y -c conda-forge -n $env_name tensorflow
conda install -y -c rdkit -n $env_name rdkit


# Run the setup.py script
echo "Install the $env_name package"
pip install -e .
# conda clean -y --tarballs
