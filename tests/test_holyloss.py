import torch
import unittest as ut
from perspectron_eai.base.perspectron import hloss_from_agreement_matrix


def loopy_hloss(agreement_matrix:torch.Tensor):
    n = agreement_matrix.size(0)
    left, right = 0, 0
    for i in range(agreement_matrix.size(0)):
        for j in range(agreement_matrix.size(1)):
            if i == j:
                left += agreement_matrix[i, j]
            else:
                right += torch.exp(agreement_matrix[i, j])
    left = (-1/n)*left
    right = torch.log(right/(n*(n-1)))
    return left + right


class TestHloss(ut.TestCase):
    def test_hloss(self):
        A = torch.rand((10, 10))
        expected = loopy_hloss(A)
        res = hloss_from_agreement_matrix(A)
        self.assertAlmostEqual(float(expected.data.numpy()), float(res.data.numpy()), places=5)