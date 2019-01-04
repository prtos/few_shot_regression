from .cnn import ClonableModule, Cnn1dFeaturesExtractor
from .fc import FcFeaturesExtractor
from .lstm import LstmFeaturesExtractor
from .tcnn import TcnnFeaturesExtractor
from .graph_cnn import GraphCnnFeaturesExtractor


class FeaturesExtractorFactory:
    name_map = dict(
        tcnn=TcnnFeaturesExtractor,
        cnn=Cnn1dFeaturesExtractor,
        lstm=LstmFeaturesExtractor,
        fc=FcFeaturesExtractor,
        gcnn=GraphCnnFeaturesExtractor)

    def __init__(self):
        super(FeaturesExtractorFactory, self).__init__()

    def __call__(self, arch:str, **kwargs):
        if arch not in self.name_map:
            raise Exception(f"Unhandled feature extractor. The name of \
             the architecture should be one of those: {list(self.name_map.keys())}")
        fe_class = self.name_map[arch.lower()]
        feature_extractor = fe_class(**kwargs)
        return feature_extractor


if __name__ == "__main__":
    factory = FeaturesExtractorFactory()
    factory(arch='fc', input_size=100, hidden_sizes=200, normalize_features=True)