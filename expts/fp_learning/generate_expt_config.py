import json
import argparse
from pprint import pprint
from configs import ConfigFactory


def main(datasets, algos, config_name, use_graph=True):
    algo_grids = []
    for dataset in datasets:
        algo_dict = ConfigFactory()(dataset, use_graph=use_graph)
        for a in algos:
            p = algo_dict[a]
            if isinstance(p, list):
                algo_grids += p
            elif isinstance(p, dict):
                algo_grids.append(p)
            else:
                raise Exception("Algo parameter must be a list or a dict")

    a_temp = '_'.join(algos)
    b_temp = '_'.join(datasets)
    if (config_name is not None):
        fname = config_name
    else:
        fname = f'config_{b_temp}_{a_temp}.json'
    with open(fname, 'w') as fd:
        json.dump(algo_grids, fd, indent=2, sort_keys=True)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datasets', default=['easytoy'], nargs='+',
    #                     type=str, help='Names of the datasets for the experiments')
    # parser.add_argument('--algos',
    #                     default=['mars'],
    #                     type=str, nargs='+',
    #                     help='Names of the algos tested')
    # parser.add_argument('--outfile', default=None,
    #                     type=str, help='Output file name')
    # args = parser.parse_args()
    # algos, datasets, config_name = args.algos, args.datasets, args.outfile

    main(['pubchemtox'], ['metakrr_mk', 'metakrr_sk'], 'config_pubchem.json', use_graph=False)
