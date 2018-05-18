from main import *

def comparison_expts(algo, arch, dataset_name, k_per_episode,
                     fit_params, eval_params, expts_dir, max_episodes=25000,
                     nb_examples_per_epoch=20000,
                     batch_size=64, fit=True, fold=None):
    n_epochs = 1
    steps_per_epoch = int(nb_examples_per_epoch/(batch_size*k_per_episode))
    data_iterator = load_data(dataset_name, k_per_episode, batch_size, fold=fold)

    expts_dir = SAVING_DIR_FORMAT.format(expts_dir=expts_dir, algo=algo,
                                         arch=arch, dataset_name=dataset_name)
    if not os.path.exists(expts_dir):
        os.makedirs(expts_dir, exist_ok=True)

    for meta_train, meta_valid, meta_test in data_iterator:
        res = get_outfile_names(expts_dir, k_per_episode, fit_params, fold)
        log_fname, ckp_fname, result_fname = res
        if hasattr(meta_train, 'ALPHABET_SIZE'):
            fit_params.update(dict(vocab_size=meta_train.ALPHABET_SIZE))
        metalearner = get_model(algo, arch, fit_params)
        if fit:
            metalearner.fit(meta_train, meta_valid, steps_per_epoch=steps_per_epoch,
                            n_epochs=n_epochs, max_episodes=max_episodes, batch_size=batch_size,
                            log_filename=log_fname, checkpoint_filename=ckp_fname)
        else:
            metalearner.load(ckp_fname)
            metalearner.network

        if eval_params is None:
            scores = metalearner.evaluate(meta_test)
            with open(result_fname, "w") as outfile:
                results = save_stats(scores, outfile)
                print(results)
        else:
            with open(result_fname, "w") as outfile:
                for p in ParameterGrid(eval_params):
                    print(p)
                    scores = metalearner.evaluate(meta_test, **p)
                    outfile.write(dict2str(p)+'\n')
                    results = save_stats(scores, outfile)
                    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', default=0, type=int, help='the number of the job run in this execution')
    parser.add_argument('--nb_jobs', default=-1, type=int, help='The numbers of job scheduled. -1 means all the jobs')
    parser.add_argument('--dataset', default='movielens', type=str, help='The name of the dataset for the experiments')
    parser.add_argument('--outdir', default='expt_results', type=str, help='The name of the output directory for the experiments')
    parser.add_argument('--algos',
                        default=['krr', 'snail', 'mann', 'maml', 'pretrain'],
                        type=str, nargs='+',
                        help='The name of the algos: krr|snail|mann|maml|pretrain')
    args = parser.parse_args()

    algos, part, nb_jobs, dataset = args.algos, args.part, args.nb_jobs, args.dataset
    if dataset == 'mhc':
        from config_mhc import *
    elif dataset == 'bindingdb':
        from config_bdb import *
    elif dataset == 'movielens':
        from config_movielens import *

    algo_dict = {'krr': grid_krr, 'mann': grid_mann, 'maml': grid_maml,
                 'snail': grid_snail, 'pretrain': grid_pretrain}
    algo_grids = [algo_dict[a] for a in algos]

    magic_number = 42
    np.random.seed(magic_number)
    torch.manual_seed(magic_number)
    params_list = list(ParameterGrid(algo_grids))
    np.random.shuffle(params_list)
    nb_jobs = len(params_list) if nb_jobs == -1 else nb_jobs
    if len(params_list) % nb_jobs == 0:
        nb_per_part = int(len(params_list)/nb_jobs)
    else:
        nb_per_part = int(len(params_list)/nb_jobs) + 1
    start_index, end_index = part*nb_per_part, (part+1)*nb_per_part
    if start_index > len(params_list):
        exit()
    print(start_index, end_index)
    params_for_this_part = params_list[start_index: end_index]

    expts_directory = os.path.join(args.outdir, dataset)
    for param in params_for_this_part:
        print(param)
        comparison_expts(**param, expts_dir=expts_directory, nb_examples_per_epoch=nb_examples_per_epoch,
                         batch_size=batch_size, max_episodes=max_episodes, fit=False)