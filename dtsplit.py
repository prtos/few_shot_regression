import click
import os
from sklearn.model_selection import train_test_split
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import glob

@click.group()
def cli():
    pass

@cli.command()
@click.argument('path')
@click.option('-ext', help='Extension of the files', type=str, default=".tsv")
@click.option('-n', '--maxN', help="Maximum number of tasks to select", type=int)
@click.option('-out', '--outfile', help='Where to save the splited dataset')
@click.option('--seed', help='Seed for spliting the dataset', type=int, default=42)
@click.option('--test_size', help="Size of the testing set", type=float, default=0.25)
def build(path, ext, maxn, outfile, seed, test_size):
    files = [os.path.join(par, f) for par, _, fn in os.walk(os.path.expanduser(path)) for f in fn if f.endswith(ext)][:maxn]
    np.random.seed(seed)
    if (test_size > 1.0) or (test_size < 0):
        raise ValueError("test_size expected to be in range [0, 1], got {}".format(test_size))
    train_files, test_files = train_test_split(files, test_size=test_size)
    with open(outfile, 'w') as OUT:
        json.dump({'Dtrain':train_files, 'Dtest':test_files}, OUT)

@cli.command()
@click.argument('path')
@click.option('-o', '--outfile', help="Output file for the figure", type=str, default="output.pdf")
@click.option('-ext', help='Extension of the files', type=str, default=".csv")
def pubchemdist(path, outfile, ext):
    sns.set_style('white')
    np.random.seed(5)
    pbdir = np.random.permutation([x for x in next(os.walk(path))[1] if x!='aids'])
    dir2size = {}
    for d in pbdir:
        nfiles = 0
        try:
            flist = os.path.join(path, d, 'metadata.txt')
            nfiles = sum(1 for line in open(flist) if line.strip())
        except:
            flist = glob.glob(os.path.join(path, d, '*'+ext))
            nfiles = len(flist)
        dir2size[d] =  nfiles

    labels, data = zip(*dir2size.items())
    percent = np.array(data)*100 / np.sum(data)
    colors = sns.husl_palette(len(labels), s=.8, l=0.7)
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(aspect="equal"))
    explode = [0.002]*len(labels)
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.3, linewidth=5), startangle=40, colors=colors, explode=explode)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), zorder=0, va="center", size=8)

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2.0 + p.theta1 # middle of the arrow
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i]+ ": {:.2f}%".format(percent[i]), xy=(x, y), xytext=(1.2*np.sign(x), 1.3*y),
                     horizontalalignment=horizontalalignment, **kw)
    plt.tight_layout()
    fig.savefig(outfile, dpi=600)

if __name__ == '__main__':
    cli()



