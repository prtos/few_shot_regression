import os
import tarfile
from ivbase.utils.datacache import DataCache
import click
import shutil

def extract_all_model_archives(dir_path, rem_fail=False):
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            dir_name = fpath[:-7]
            if fname.endswith('.tar.gz'):
                tar = tarfile.open(fpath, 'r:gz')
                tar.extractall(path=dir_name, )
                tar.close()
            if os.path.isdir(dir_name):
                if rem_fail and os.path.exists(os.path.join(dir_name, 'failure')):
                    shutil.rmtree(dir_name, ignore_errors=True)

@click.group()
def cli():
    pass

@cli.command()
@click.option('-s3', '--s3_path', default='iscb-expts', help="S3 location of the experiment results (model artifacts and output files) are saved")
@click.option('-o', '--output_path', help="Local path where the data is saved to.")
@click.option('-nf', '--no_failure', is_flag=True, help="Do not save experiments with failure.")
def get_result(s3_path, output_path, no_failure):
    cache = DataCache(cache_root=output_path)
    s3_path = os.path.join('s3://invivoai-sagemaker-artifacts/', s3_path)
    local_path = cache.sync_dir(s3_path)
    extract_all_model_archives(local_path, no_failure)

if __name__ == '__main__':
	cli()