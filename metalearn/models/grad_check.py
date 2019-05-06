"""
This part of the code is heavily inspired by the snippet provided by Adam Paszke (now canonical)
which can be found here: https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
Copyright (c) 2017 - 2019, the respective contributors.
"""
import os
import glob
import torch
import logging
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.image as mpimg
from matplotlib.animation import FFMpegWriter
from graphviz import Digraph, Source
from pytoune.framework.callbacks import Callback
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from collections import defaultdict, OrderedDict
logger = logging.getLogger("GradInspect")
logger.setLevel(logging.DEBUG)
matplotlib.use('agg')


def iter_graph(root, callback_fn):
    """Iter through a computational graph and run callback on each node encountered"""
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback_fn(fn)


class GradFlow(Callback):
    r"""
    Analysis of Gradient Flow in a pytorch network

    Arguments
    ---------
        check_nan: bool, optional
            Whether to check nan gradient
            (Default value = True)
        check_zero bool, optional
            Whether to check zero gradient (Use to locate vanishing gradient issues)
            (Default value = True)
        check_explod: bool, optional
            Whether to check exploding gradient
            (Default value = True)
        last_only: bool, optional
            Whether to log the gradient flow during the last batch only
            (Default value = True)
        write_grad: bool, optional
            Whether to log gradients in tensorboard x if the writer is available.
            (Default value = False)
        outfile: str, optional
            Output file where to save a drawing of the gradient flow. Nothing is saved by default.
            (Default value = None)
        enforce_sanity: bool, optional
            Ensure that the model computational graph does not change during backprop
            (Default value = False)

    Attributes
    ----------
        dots: list
            All saved computational graph with gradient flow during training.
        outfile: str or None
            Output file where to save the graph drawing
        sanity: bool
            Whether sanity is enforced.
    """
    NODE_COLOR = {"bad": "#211D1E", "nan": "#FF9191",
                  "large": "#56B6FF", "accum": "#62E0AB", "zero": "#C8CBCE"}

    def __init__(self, check_nan=True, check_zero=True, check_explod=True, last_only=True, write_grad=False, max_val=1e6, outfile=None, enforce_sanity=False, save_every=1):
        super().__init__()
        self.check_nan = check_nan
        self.check_zero = check_zero
        self.check_explod = check_explod
        self.last_only = last_only
        self.max_val = max_val
        self.dots = []
        self.hook = None
        self.log_grad = write_grad
        self.outfile = outfile
        self.sanity = enforce_sanity
        self.save_every = save_every

    def _add_legend(self, dot=None):
        """Add legend to a digraph plot"""
        node_attr = dict(style='filled',
                         shape='box',
                         color="#333333",
                         align='center',
                         fontsize='7',
                         height='0.2'
                         )

        legend = Digraph(node_attr=node_attr, graph_attr=dict(
            size="4,8", splines="ortho"))
        legend.name = "cluster_legend"
        legend.node("accum", "Grad Accum",
                    fillcolor=self.NODE_COLOR["accum"], shape="box")
        legend.node("zero", "Zero Grad",
                    fillcolor=self.NODE_COLOR["zero"], shape="hexagon")
        legend.node("expl", "Large Grad",
                    fillcolor=self.NODE_COLOR["large"], shape="box")
        legend.node("nan", "NaN Grad",
                    fillcolor=self.NODE_COLOR["nan"], shape="box")
        legend.node("bad", "Bad Grad",
                    fillcolor=self.NODE_COLOR["bad"], fontcolor="white", shape="ellipse")
        legend.edge("accum", "zero", style="invis")
        legend.edge("zero", "expl", style="invis")
        legend.edge("expl", "nan", style="invis")
        legend.edge("nan", "bad", style="invis")
        legend.subgraph(dot)
        return legend

    def on_backward_start(self, batch, loss=None):
        """
        This method is called after the loss computation but before the backpropagation and optimization step

        Arguments
        ---------
            batch: int
                The batch number.
            loss: `torch.FloatTensor`

        """
        if self.save_every % batch == 0:
            self.hook = self.register_hooks(loss)

    def on_backward_end(self, batch):
        """
        Is called after the backprop but before the optimization step

        Arguments
        ----------
            batch : int
                The batch number

        """
        if callable(self.hook):
            graph = self.hook()
            if self.last_only:
                self.dots = [graph]
            else:
                self.dots.append(graph)

        model = getattr(self, "model", None)
        if self.log_grad and model and model.writer:
            for name, param in model.model.named_parameters():
                if param.grad is not None and name:
                    model.writer.add_histogram(
                        "grad/{}".format(name), param.data.cpu(), batch)
                    model.writer.add_histogram(
                        "weights/{}".format(name), param.grad.data.cpu(), batch)

    def on_batch_end(self, batch, logs):
        """
        Called before the end of each batch

        Args:
            batch (int): The batch number.
            logs (dict): Contains the following keys:

                 * 'batch': The batch number.
                 * 'loss': The loss of the batch.
                 * Other metrics: One key for each type of metrics.

        Example::

            logs = {'batch': 6, 'loss': 4.34462, 'accuracy': 0.766}
        """
        super().on_batch_end(batch, logs)
        model = getattr(self, "model", None)

        if self.outfile and len(self.dots) > 0:
            outfile = self.outfile.format(batch)
            if model is not None and hasattr(model, 'model_dir'):
                outfile = os.path.join(model.model_dir, outfile)
            self._add_legend(self.dots[-1]).save(outfile)
            Source.from_file(outfile, format='svg').render()

    def is_zero(self, grad_output):
        """
        Check if gradient is zero. This can be helpfull when trying to
        identify vanishing gradient problem (zero everywhere).
        """
        grad_output = grad_output.data
        return grad_output.allclose(torch.zeros_like(grad_output), equal_nan=True)

    def is_large(self, grad_output):
        """
        Check exploding gradient
        """
        grad_output = grad_output.data
        return grad_output.gt(self.max_val).any()

    def is_nan(self, grad_output):
        """
        Check nan gradient.
        """
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any()

    def is_bad_grad(self, grad_output):
        """
        Check if gradient is bad (nan or extremely large
        """
        return self.is_large(grad_output) or self.is_nan(grad_output)

    def register_hooks(self, var):
        """Register a hook on the loss function to record gradient flow"""
        cur_instance = self
        fn_dict = {}

        def hook_cb(fn):
            def register_grad(grad_input, grad_output):
                fn_dict[fn] = grad_input
            fn.register_hook(register_grad)

        iter_graph(var.grad_fn, hook_cb)

        def make_graph():
            node_attr = dict(style='filled',
                             shape='box',
                             color="#333333",
                             align='left',
                             fontsize='12',
                             ranksep='0.1',
                             height='0.2')
            dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

            def size_to_str(size):
                return '(' + (', ').join(map(str, size)) + ')'

            def build_graph(fn):
                if hasattr(fn, 'variable'):  # if GradAccumulator
                    u = fn.variable
                    node_name = 'Variable\n ' + size_to_str(u.size())
                    dot.node(str(id(u)), node_name,
                             fillcolor=cur_instance.NODE_COLOR["accum"])
                else:
                    if self.sanity and fn not in fn_dict:
                        err_fn = fn
                        queue = [fn]
                        seen = set()
                        while queue:
                            fn = queue.pop()
                            if fn in seen:
                                continue
                            seen.add(fn)
                            for next_fn, _ in fn.next_functions:
                                if next_fn is not None:
                                    queue.append(next_fn)
                                    next_id = id(
                                        getattr(next_fn, 'variable', next_fn))
                                    if next_fn in fn_dict:
                                        dot.node(str(next_id), str(
                                            type(next_fn).__name__), fillcolor="white")
                                    else:
                                        dot.node(str(next_id), str(
                                            type(next_fn).__name__))
                                    dot.edge(str(next_id), str(id(fn)))
                        dot.save(self.outfile + ".err")
                        raise AssertionError(err_fn, id(err_fn))

                    val = dict(fillcolor='white', shape="box")
                    if cur_instance.check_zero and all(cur_instance.is_zero(gi) for gi in fn_dict.get(fn, [None]) if gi is not None):
                        val.update(
                            fillcolor=cur_instance.NODE_COLOR["zero"], shape="hexagon")
                    else:
                        both_bad = 0
                        if cur_instance.check_nan and any(cur_instance.is_nan(gi) for gi in fn_dict.get(fn, [None]) if gi is not None):
                            val.update(
                                fillcolor=cur_instance.NODE_COLOR["nan"])
                            both_bad += 1
                        if cur_instance.check_explod and any(cur_instance.is_large(gi) for gi in fn_dict.get(fn, [None]) if gi is not None):
                            val.update(
                                fillcolor=cur_instance.NODE_COLOR["large"])
                            both_bad += 1
                        if both_bad == 2:
                            val.update(
                                fontcolor="white", fillcolor=cur_instance.NODE_COLOR["bad"], shape="ellipse")
                    dot.node(str(id(fn)), str(type(fn).__name__), **val)

                for next_fn, _ in fn.next_functions:
                    if next_fn is not None:
                        next_id = id(getattr(next_fn, 'variable', next_fn))
                        dot.edge(str(next_id), str(id(fn)))

            iter_graph(var.grad_fn, build_graph)
            return dot

        return make_graph


class GradientInspector(Callback):
    r"""
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    and to visualize the gradient flow.

    Arguments
    ---------
        bottom_zoom: float, default=-0.001
            Min value of the gradient to be shown
        top_zoom: float, default=0.02
            Max value of the gradient to be shown. Use this conjointly with the bottom_zoom to zoom into
            only part of the plot. Please keep in mind that only the abs value of the gradient matter.
        max_norm: float, default=4
            Max range of the gradient norm to display
        speed: float, default=0.05
            Time to wait before updating the plot (in seconds)
        log: bool, default=False
            Whether to print in terminal
        update_at: str, default="backward"
            When should the plot be updated ? One of {backward, epoch, batch}.
        figsize: tuple, default=(10, 6)
            Size of the plot
        showbias: bool, default=False
            Whether bias should be considered
        outfile: str, default=None
            Path to save the animation during the whole training
    """

    def __init__(self, bottom_zoom=-0.001, top_zoom=0.02, max_norm=4, speed=0.05, log=False, update_at="backward", figsize=(10, 6), showbias=False, outdir=None, clean_temps=False):
        super().__init__()
        self.fig, self.axes = plt.subplots(ncols=2, sharey=True, tight_layout=True, gridspec_kw={
                                           'width_ratios': [3, 1]}, figsize=figsize)
        self.speed = speed
        self.top_zoom = top_zoom
        self.bottom_zoom = bottom_zoom
        self.updater = update_at
        self.bias = showbias
        self.max_norm = max_norm
        self.outdir = os.path.abspath(outdir)
        self.figsize = figsize
        os.makedirs(self.outdir, exist_ok=True)
        self.outfile = os.path.join(self.outdir, 'summumary_video.mp4')
        self.saving_n = 0
        self.clean = clean_temps
        self.folder_clean()
        # plt.ioff()

    def _configure_logger(self, log):
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        if log:
            ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def log_grad(self, params, header="Epoch", padding=0):
        logger.debug("\t" * padding + "==> " + header)
        for name, param in params.items():
            # if param.grad is not None:
            # logger.debug("\t"*padding + f"{name}: No grad")
            logger.debug("\t" * padding +
                         "{}: {}".format(name, param.abs().mean()))

    def on_epoch_begin(self, epoch, logs=None):
        self.parameters_means = defaultdict(list)

    def on_batch_end(self, batch, logs=None):
        grads = OrderedDict()
        for name, param in self.model.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.data.cpu().clone()
                self.parameters_means[name].append(grads[name])
        self.log_grad(grads, "Batch: {}".format(batch), padding=1)
        self.update(grads)
        # if self.updater == "batch":
        #     try:
        #         self.update(grads)
        #     except KeyboardInterrupt:
        #         raise
        #     except:
        #         pass

    def on_epoch_end(self, epoch, logs=None):
        grads = OrderedDict((name, torch.stack(params))
                            for name, params in self.parameters_means.items())
        self.log_grad(grads, "Epoch")
        self.update(grads)
        self.to_video(self.outfile, interval=100, clean=False)
        # if self.updater == "epoch":
        #     try:
        #         self.update(grads)
        #     except KeyboardInterrupt:
        #         raise
        #     except:
        #         pass

    def on_backward_end(self, batch):
        r"""Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        and to visualize the gradient flow"""
        if self.updater == "backward":
            grads = OrderedDict((name, param.grad.data.cpu(
            )) for name, param in self.model.model.named_parameters() if param.grad is not None)
            try:
                self.update(grads)
            except KeyboardInterrupt:
                raise
            except:
                pass

    def __del__(self):
        """Force saving to video before closing"""
        self.to_video(outfile=self.outfile, clean=True)
        plt.close(self.fig)

    def _clear_axis(self):
        """Clear the axis of the figure"""
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.axis('off')

    def save(self):
        self.fig.savefig(os.path.join(self.outdir, "temp_{}.png".format(self.saving_n)), dpi=150)
        self.saving_n += 1

    def folder_clean(self):
        files = glob.glob(os.path.join(self.outdir, "temp_*.png"))
        for f in files:
            os.unlink(f)

    def to_video(self, outfile, interval=100, clean=False):
        fig = plt.figure(figsize=self.figsize)
        plt.axis("off")
        if os.path.exists(outfile):
            os.unlink(outfile)
        files = [os.path.join(self.outdir, "temp_{}.png".format(i))
                 for i in range(self.saving_n)]
        files = files[-50:] if len(files) > 50 else files

        images = [[plt.imshow(mpimg.imread(f), animated=True)] for f in files]
        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True,
                                        repeat_delay=1000)
        ani.save(outfile, writer=writer)
        if clean:
            self.folder_clean()

    def update(self, grads):
        ave_grads = []
        max_grads = []
        layers = []
        norm_grads = []

        for ax in self.axes.flat:
            ax.clear()

        for n, p in grads.items():
            if ("bias" not in n or self.bias):
                layers.append(n + " (" + ",".join(str(x) for x in p.shape) + ")")
                ave_grads.append(p.abs().mean())
                max_grads.append(p.abs().max())
                norm_grads.append(p.norm())
        self.axes[0].barh(np.arange(len(max_grads)), max_grads,
                          height=0.6, zorder=10, alpha=0.4, lw=1, color="#A3D6D2")
        self.axes[0].barh(np.arange(len(max_grads)), ave_grads,
                          height=0.6, zorder=10, alpha=0.9, lw=1, color="#2F6790")
        self.axes[0].vlines(0, -0.5, len(ave_grads) + 1, lw=2, color="k")
        self.axes[1].barh(np.arange(len(max_grads)), norm_grads,
                          height=0.6, zorder=10, alpha=0.5, color="gray")
        self.axes[0].invert_xaxis()
        self.axes[0].set(title='Gradient values')
        self.axes[1].set(title='Gradient norm')
        self.axes[0].set_ylim(bottom=-0.5, top=len(ave_grads))
        # zoom in on the lower gradient regions
        self.axes[0].set_xlim(right=self.bottom_zoom, left=self.top_zoom)
        self.axes[0].yaxis.set_tick_params(
            left=False, labelright=False, labelleft=False)
        self.axes[1].yaxis.set_tick_params(
            labelright=True, labelleft=False, left=False, right=True, labelsize=9)
        self.axes[1].set(yticks=range(0, len(ave_grads)), yticklabels=layers)
        # zoom in on the lower gradient regions
        self.axes[1].set_xlim(left=0, right=self.max_norm)

        for ax in self.axes.flat:
            ax.margins(0.01)
            ax.grid(True)

        self.fig.legend([Line2D([0], [0], color="#A3D6D2", lw=3),
                         Line2D([0], [0], color="#2F6790", lw=3), Line2D([0], [0], color="gray", lw=3)], ['max-gradient', 'mean-gradient', "norm"], loc='upper right', bbox_to_anchor=(1, 1), fontsize=9)
        self.fig.subplots_adjust(wspace=0.023)
        self.save()
        self._clear_axis()
