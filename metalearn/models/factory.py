from .base import *
from .maml import MAML
from .mann import MANN
from .snail import SNAIL
from .metakrr_singlekernel import MetaKrrSingleKernelLearner
from .metargr_multikernels import MetaKrrMultiKernelsLearner, MetaGPMultiKernelsLearner
from .metagp_singlekernel import MetaGPSingleKernelLearner
from .deep_prior import DeepPriorLearner
from .metarf import MetaRFLearner
from .metaboost import MetaBoostLearner
from .mars import MarsLearner

class ModelFactory:
    name_map = dict(
        mann=MANN,
        maml=MAML,
        snail=SNAIL,
        deep_prior=DeepPriorLearner,
        metakrr_sk=MetaKrrSingleKernelLearner,
        metakrr_mk=MetaKrrMultiKernelsLearner,
        metagp_sk=MetaGPSingleKernelLearner,
        metagp_mk=MetaGPMultiKernelsLearner,
        metarf=MetaRFLearner,
        metaboost=MetaBoostLearner,
        mars=MarsLearner,
    )

    def __init__(self):
        super(ModelFactory, self).__init__()

    def __call__(self, model_name:str, **kwargs):
        if model_name not in self.name_map:
            raise Exception(f"Unhandled model. The name of \
             the model should be one of those: {list(self.name_map.keys())}")
        modelclass = self.name_map[model_name.lower()]
        model = modelclass(**kwargs)
        return model


if __name__ == "__main__":
    factory = ModelFactory()
    factory(arch='fc', input_size=100, hidden_sizes=200, normalize_features=True)