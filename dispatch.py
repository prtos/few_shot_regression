import sys, os, argparse, subprocess
from sklearn.model_selection import ParameterGrid
from few_shot_regression.expts_utils import get_config_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', default='local', type=str,
                        help='The name of the server: local|graham|helios')
    parser.add_argument('-o', '--outdir', default='results/test', type=str,
                        help='The name of the output directory for the experiments')
    parser.add_argument('-a', '--algos',
                        default=['multitask'],  # ['fskrr', 'metakrr', 'multitask', 'snail', 'mann', 'maml'],
                        type=str, nargs='+',
                        help='The name of the algos: fskrr|metakrr|multitask|snail|mann|maml')
    parser.add_argument('-d', '--dataset',
                        default='bindingdb',
                        type=str,
                        help='The name of the dataset: mhc|bindingdb|movielens')
    args = parser.parse_args()
    server, algos, dataset = args.server, args.algos, args.dataset
    expts_directory = args.outdir
    if not os.path.exists(expts_directory):
        os.makedirs(expts_directory, exist_ok=True)

    algo_dict, _ = get_config_params(dataset)
    algo_grids = [algo_dict[a] for a in algos]
    algos_str = ' '.join(algos)
    ntasks = len(list(ParameterGrid(algo_grids)))
    print("Quick summary")
    print('System:', server)
    print('Dataset:', dataset)
    print('Algos:', algos_str)
    print("Experiment folder:", args.outdir, expts_directory)
    print('Number of tasks dispatched:', ntasks)

    if server == "local":
        c = "~/anaconda3/bin/python main.py --outdir {out} --algos {algos} --dataset {dataset}".format(
            out=expts_directory, algos=algos_str, dataset=dataset)
        c = os.path.expanduser(c)
        subprocess.run(c.split(' '))
    else:
        if server == 'graham':
            launcher, prototype = 'sbatch', "submit_graham.sh"
        elif server == 'helios':
            launcher, prototype = 'msub', "submit_helios.sh"
        else:
            launcher, prototype = None, None
            Exception("Server {} is not found".format(server))

        with open(prototype) as f:
            content = f.read()
            content = content.format(out=expts_directory, algos=algos_str, ntasks=ntasks, dataset=dataset)
        os.chdir(os.path.expanduser(expts_directory))
        with open(prototype, 'w') as f:
            f.write(content)
        subprocess.Popen([launcher, prototype])

