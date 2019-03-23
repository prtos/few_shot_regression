from .base import *
from .maml import MAML
from .mann import MANN
from .snail import SNAIL
from .metakrr_mk import MetaKrrMKLearner
from .metakrr_sk import MetaKrrSKLearner
from .fp_learner import FPLearner
from .seq2seq_fp import Seq2SeqLearner
from .cnp import CNPLearner


class ModelFactory:
    name_map = dict(
        mann=MANN,
        maml=MAML,
        snail=SNAIL,
        metakrr_sk=MetaKrrSKLearner,
        metakrr_mk=MetaKrrMKLearner,
        fingerprint=FPLearner,
        seqtoseq=Seq2SeqLearner,
        cnp=CNPLearner,
    )

    def __init__(self):
        super(ModelFactory, self).__init__()

    def __call__(self, model_name, **kwargs):
        if model_name not in self.name_map:
            raise Exception(f"Unhandled model. The name of \
             the model should be one of those: {list(self.name_map.keys())}")
        modelclass = self.name_map[model_name.lower()]
        model = modelclass(**kwargs)
        return model


if __name__ == "__main__":
    factory = ModelFactory()
    model = factory(arch='fc', input_size=100, hidden_sizes=200, normalize_features=True)
