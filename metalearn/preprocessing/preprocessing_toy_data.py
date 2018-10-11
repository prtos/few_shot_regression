import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_FOLDER = os.path.join('../../datasets/toy_easy')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER, exist_ok=True)
MAX_DATASIZE = int(5e3)


def periodic_fn(base_freq, amplitudes, phases, instruments):
    """Returns a periodic function composed of different harmonics at different phases."""
    def fn(t):
        components = []
        for i, (A, phi, instrument) in enumerate(zip(amplitudes, phases, instruments)):
            components.append(1 * instrument(t*base_freq*float(i+1) + phi))
        return np.sum(components, 0)
    return fn


def random_periodic_fn(base_freq_range=(1.,1.), n_harmonics=2, fix_base_magnitude=False, rng=np.random):
    base_freq = rng.uniform(*base_freq_range)
    # instruments = rng.choice((np.sin, signal.sawtooth), n_harmonics)
    instruments = rng.choice((np.sin, ), n_harmonics)

    magnitudes = rng.randn(n_harmonics)
    if fix_base_magnitude:
        magnitudes[0] = 1.
    return periodic_fn(base_freq, magnitudes, rng.rand(n_harmonics) * 2 * np.pi, instruments)


def make_dataset(fn, start, stop, n, normal_domain, noise, rng):
    if normal_domain:
        std = (stop - start)/4.
        mu = (start + stop)/2.
        x = (rng.randn(n) * std + mu)
    else:
        x = (rng.rand(n) * (stop - start) + start)
    x = x.flatten()
    x = x[np.argsort(x)]
    y = (fn(x) + rng.randn(*x.shape) * noise).flatten()
    return x, y


def make_meta_dataset(n_functions, base_freq_range=(1, 10), max_harmonics=20, rng=np.random, fname_prefix=''):
    for i in range(n_functions):
        n_harmonics = 1 + rng.choice(max_harmonics)
        magnitudes = rng.randn(n_harmonics)
        instruments = rng.choice((np.sin, np.cos), n_harmonics)
        base_freq = rng.uniform(*base_freq_range)
        limit = 3
        noise = rng.rand() * 0.033
        fn = periodic_fn(base_freq, magnitudes, rng.rand(n_harmonics) * 2 * np.pi, instruments)
        x, y = make_dataset(fn, start=-limit, stop=limit, n=1000, normal_domain=False, noise=noise, rng=rng)
        metadata = pd.DataFrame(dict(n_harmonics=[n_harmonics],
                                     base_freq=[base_freq],
 #                                    magnitudes=[magnitudes],
                                     instruments=[instruments],
                                     ))
        data = pd.DataFrame(data=dict(x=x, y=y))
        fname = os.path.join(DATA_FOLDER, '{}_function_{}.csv'.format(fname_prefix, i))
        metadata.to_csv(fname, index=False)
        data.to_csv(fname, index=False, mode='a')
        x, y = make_dataset(fn, start=-limit, stop=limit, n=50000, normal_domain=True, noise=noise, rng=rng)
        fname = os.path.join(DATA_FOLDER, '{}_function_{}.png'.format(fname_prefix, i))
        plt.figure()
        plt.plot(x, y)
        plt.savefig(fname)
        plt.close()


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    make_meta_dataset(5000, base_freq_range=(1, 5), max_harmonics=2, rng=rng, fname_prefix='')
    # make_meta_dataset(4000, base_freq_range=(1, 5), max_harmonics=6, rng=rng, fname_prefix='train')
    # make_meta_dataset(1000, base_freq_range=(1, 5), max_harmonics=6, rng=rng, fname_prefix='valid')
    # make_meta_dataset(1000, base_freq_range=(1, 5), max_harmonics=6, rng=rng, fname_prefix='test')

    # n_harmonics = 2
    # fn = random_periodic_fn(base_freq_range=(0, 0), n_harmonics=n_harmonics, fix_base_magnitude=False)
    # x, y = make_dataset(fn, -5, 5, 1000, True, 0, np.random)
    # x, y = x.flatten(), y.flatten()
    # ti = np.argsort(x)
    # x, y = x[ti], y[ti]
    # print(x.shape, y.shape)
    # plt.plot(x, y)
    # plt.show()
