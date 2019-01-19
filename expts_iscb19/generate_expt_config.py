import json
import argparse
from pprint import pprint
from configs import ConfigFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='easytoy', type=str, help='The name of the dataset for the experiments')
    parser.add_argument('--algos',
                        default=['mars'],
                        type=str, nargs='+',
                        help='The name of the algos tested')
    args = parser.parse_args()
    algos, dataset = args.algos, args.dataset

    algo_dict = ConfigFactory()(dataset)
    algo_grids = [algo_dict[a] for a in algos]

    a_temp = '_'.join(algos)
    fname = f'config_{dataset}_{a_temp}.json'
    with open(fname, 'w') as fd:
        json.dump(algo_grids, fd, indent=2, sort_keys=True) 
