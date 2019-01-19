import unittest as ut
import numpy as np
import warnings


def gauss_pdf(z, mu, std):
    return np.exp(-0.5*((z - mu)/std)**2)/(std*np.sqrt(2*np.pi))


def compute_mu_std_z_proba_trio(mu_a, mu_b, mu_z, std_a, std_b, std_z):
    deno = (std_a * std_z) ** 2 + (std_b * std_z) ** 2 - (std_a * std_b) ** 2
    num = (mu_b * (std_a * std_z) ** 2) + (mu_a * (std_b * std_z) ** 2) - (mu_z * (std_a * std_b) ** 2)
    mu = num / deno
    std = np.sqrt(((std_a * std_b * std_z) ** 2) / deno)

    Z = (std_a*std_b*np.sqrt(2*np.pi)) / std_z
    Z *= np.exp((0.5 * (mu_a / std_a)**2)
                + (0.5 * (mu_b / std_b) ** 2)
                - (0.5 * (mu_z / std_z) ** 2)
                - (0.5 * (mu / std)**2)
                )
    return mu, std, Z


def proba_trio_exists(mu_a, mu_b, mu_z, std_a, std_b, std_z):
    return std_z >= np.sqrt(((std_a*std_b)**2) / ((std_a**2) + (std_b**2)))


class TestAgreementNormalDistributions(ut.TestCase):
    def test_without_Z(self):
        epsilon = 1e-2
        mu_a, std_a = np.random.rand(), np.random.rand()+epsilon
        mu_b, std_b = np.random.rand(), np.random.rand()+epsilon
        mu_z, std_z = np.random.rand(), np.random.rand()+epsilon
        std_z = max(std_z, np.sqrt(((std_a*std_b)**2) / ((std_a**2) + (std_b**2)))+epsilon)
        print(std_z)

        if not proba_trio_exists(mu_a, mu_b, mu_z, std_a, std_b, std_z):
            warnings.warn('(std_a*std_z)**2 + (std_b*std_z)**2 > (std_a*std_b)**2')
        else:
            mu, std, Z = compute_mu_std_z_proba_trio(mu_a, mu_b, mu_z, std_a, std_b, std_z)

            ratios = []
            for z in np.random.rand(20):
                p_z_xa = gauss_pdf(z, mu_a, std_a)
                p_z_xb = gauss_pdf(z, mu_b, std_b)
                p_z = gauss_pdf(z, mu_z, std_z)
                expected_value = p_z_xa * p_z_xb / p_z
                approximate_value = np.exp(-0.5*((z - mu)/std)**2) / Z
                ratio = approximate_value / expected_value
                ratios.append(ratio)
            print(ratios)
            for ratio in ratios:
                self.assertAlmostEqual(ratio, 1)
