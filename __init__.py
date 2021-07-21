# from .Prachathai import PrachathaiClassificationTask
# from .Prachathai import PrachathaiBiLSTM, PrachathaiVisualCharacterBiLSTM
# from .Prachathai import PrachathaiCNN, PrachathaiWav2Vec2CNN
# from .Prachathai import PrachathaiTransformer, PrachathaiVisualCharacterTransformer
# from .Prachathai import PrachathaiEnsemble
# from .Prachathai import PrachathaiVisualText, PrachathaiVisualTextAttention
# from .Text2Speech import GoogleTTS
# from .InferencePipeline import InferencePipeline
# from .AmazonReview import AmazonReviewClassificationTask

from .SentenceClassification import BinarySentenceClassificationTrainerModule, MulticlassSentenceClassificationTrainerModule, MultilabelSentenceClassificationTrainerModule
# from .MachineTranslation import MachineTranslationModule
from .WordEmbedding import SkipgramTrainerModule, CbowTrainerModule, FastTextTrainerModule
from .WordEmbedding import WordEmbedding, CharacterLevelWordSparse, CharacterLevelWordEmbedding, PositionalCharacterLevelWordSparse, PositionalCharacterLevelWordEmbedding