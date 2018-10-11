import sys, os, argparse, subprocess, shutil
from sklearn.model_selection import ParameterGrid
from expts_utils import get_config_params

max_time_per_dataset = dict(mhc='12:00:00', bindingdb='48:00:00',
                            movielens='12:00:00', uci='48:00:00',
                            toy='24:00:00', easytoy='24:00:00')
# max_time_per_dataset = dict(mhc='0:30:00', bindingdb='0:30:00',
#                             movielens='0:30:00', uci='0:30:00')
computer_configuration = dict(uci="#SBATCH --cpus-per-task=16",
                              toy="#SBATCH --cpus-per-task=16",
                              easytoy="#SBATCH --cpus-per-task=16",
                              movieles="#SBATCH --cpus-per-task=16",
                              mhc="#SBATCH --cpus-per-task=4\n#SBATCH --gres=gpu:1",
                              bindingdb="#SBATCH --cpus-per-task=4\n#SBATCH --gres=gpu:1")

main_file = 'main.py'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', default='local', type=str,
                        help='The name of the server: local|graham|helios')
    parser.add_argument('-o', '--outdir', default='results/test', type=str,
                        help='The name of the output directory for the experiments')
    parser.add_argument('-a', '--algos',
                        default=['metakrr_mk'],  # ['fskrr', 'metakrr', 'multitask', 'snail', 'mann', 'maml'],
                        type=str, nargs='+',
                        help='The name of the algos: fskrr|metakrr|multitask|snail|mann|maml')
    parser.add_argument('-d', '--dataset',
                        default='uci',
                        type=str,
                        help='The name of the dataset: mhc|bindingdb|movielens')
    args = parser.parse_args()
    server, algos, dataset = args.server, args.algos, args.dataset
    expts_directory = args.outdir
    if not os.path.exists(expts_directory):
        os.makedirs(expts_directory, exist_ok=True)
    temp = os.path.join(expts_directory, 'configs')
    if not os.path.exists(temp):
        shutil.copytree('configs', temp)

    algo_dict = get_config_params(dataset)
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
        c = "~/anaconda3/bin/python {main_file} --outdir {out} --algos {algos} --dataset {dataset}".format(
            out=expts_directory, algos=algos_str, dataset=dataset, main_file=main_file)
        c = os.path.expanduser(c)
        subprocess.run(c.split(' '))
    else:
        if server == 'graham':
            launcher, prototype = 'sbatch', "submit_graham.sh"
        elif server == 'helios':
            launcher, prototype = 'msub', "submit_helios.sh"
        else:
            launcher, prototype = None, None
            raise Exception("Server {} is not found".format(server))

        with open(prototype) as f:
            content = f.read()
            content = content.format(out=expts_directory, algos=algos_str, main_file=main_file,
                                     ntasks=ntasks-1, dataset=dataset, time=max_time_per_dataset[dataset],
                                     computer_configuration=computer_configuration[dataset])
        os.chdir(os.path.expanduser(expts_directory))
        with open(prototype, 'w') as f:
            f.write(content)
        subprocess.Popen([launcher, prototype])

