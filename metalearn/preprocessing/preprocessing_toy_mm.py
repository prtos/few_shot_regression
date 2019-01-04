import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


DATA_FOLDER = os.path.join('../../datasets/multimodetoy')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER, exist_ok=True)
MAX_DATASIZE = int(5e3)


def periodic_fn(base_freq, amplitudes, phases, instruments):
    """Returns a periodic function composed of different harmonics at different phases."""
    def fn(t):
        components = []
        for i, (A, phi, instrument) in enumerate(zip(amplitudes, phases, instruments)):
            components.append(A * instrument(t*base_freq*float(i+1) + phi))
        return np.sum(components, 0)
    return fn


def random_periodic_fn(base_freq_range=(1.,1.), n_harmonics=2, fix_base_magnitude=False, rng=np.random):
    base_freq = rng.uniform(*base_freq_range)
    # instruments = rng.choice((np.sin, signal.sawtooth), n_harmonics)
    instruments = rng.choice((np.sin, ), n_harmonics)

    magnitudes = rng.randn(n_harmonics)
    phases = rng.rand(n_harmonics) * 2 * np.pi
    if fix_base_magnitude:
        magnitudes[0] = 1.
    return periodic_fn(base_freq, magnitudes, phases, instruments)


# def random_quadratic_fn()
def make_dataset(fn, start, stop, n, normal_domain, noise, rng):
    if normal_domain:
        std = (stop - start)/4.
        mu = (start + stop)/2.
        x = (rng.randn(n) * std + mu)
    else:
        x = (rng.rand(n) * (stop - start) + start)
    x = x.flatten()
    x = x[np.argsort(x)]
    y = (fn(x) + rng.randn(n) * noise).flatten()
    return x, y


def make_meta_dataset(n_functions, base_freq_range=(1, 10), min_harmonics=20, max_harmonics=20, rng=np.random, 
                      x_range=(-1, 1), fixed_magnitudes=True, noise=0.1, fname_prefix=''):
    assert min_harmonics <= max_harmonics

    def f(i):
        print(i)
        rng.seed(i) # important because this will called in parallel
        n_harmonics = min_harmonics + rng.choice(max_harmonics-min_harmonics+1)
        magnitudes = rng.randn(n_harmonics)
        phases = rng.rand(n_harmonics) * 2 * np.pi
        if fixed_magnitudes:
            magnitudes[0] = 1
        instruments = rng.choice((np.sin, ), n_harmonics)     # rng.choice((np.sin, np.cos), n_harmonics)
        base_freq = rng.uniform(*base_freq_range)
        fn = periodic_fn(base_freq, magnitudes, phases, instruments)
        x, y = make_dataset(fn, start=x_range[0], stop=x_range[1], n=1000, normal_domain=False, noise=noise, rng=rng)
        metadata = ([('n_harmonics',  n_harmonics)] +
                    [('base_freq', base_freq)] +
                    [('magnitudes_'+str(i), (magnitudes[i] if i < n_harmonics else 0))
                     for i in range(max_harmonics)] +
                    [('phases_' + str(i), (phases[i] if i < n_harmonics else 0))
                     for i in range(max_harmonics)] +
                    [('instruments_'+str(i), (instruments[i].__name__ if i < n_harmonics else ''))
                     for i in range(max_harmonics)])
        metadata = pd.DataFrame(data=dict(zip([el[0] for el in metadata], [[el[1]] for el in metadata])))
        data = pd.DataFrame(data=dict(x=x, y=y))
        fname = os.path.join(DATA_FOLDER, '{}_function_{}.csv'.format(fname_prefix, i))
        metadata.to_csv(fname, index=False)
        data.to_csv(fname, index=False, mode='a')
        # x, y = make_dataset(fn, start=-limit, stop=limit, n=5000, normal_domain=True, noise=noise, rng=rng)
        fname = os.path.join(DATA_FOLDER, '{}_function_{}.png'.format(fname_prefix, i))
        plt.figure()
        plt.plot(x, y)
        plt.savefig(fname)
        plt.close()

    Parallel(n_jobs=-1)(delayed(f)(i) for i in range(n_functions))


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    make_meta_dataset(5000, base_freq_range=(2, 2), x_range=(-5, 5), min_harmonics=2, max_harmonics=2, rng=rng,
        fixed_magnitudes=False, noise=0.01, fname_prefix='')
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
