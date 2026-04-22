# config\enums.py
from enum import StrEnum


class Dataset(StrEnum):
    LIBRISPEECH = "librispeech"
    BURGUNDY = "burgundy"


class TaskName(StrEnum):
    LSTM = "LSTM"
    LSTM_PHONE = "LSTM_PHONE"
    LSTM_WORD = "LSTM_WORD"
    LSTM_WORD_SEM = "LSTM_WORD_SEM"
    LAS_MOCHA_WORDS = "LAS_MOCHA_WORDS"
    LAS_WORDS = "LAS_WORDS"
    RNNT = "RNNT"


class LabelType(StrEnum):
    ONEHOT = "onehot"
    SRV = "srv"


class PolicyName(StrEnum):
    BASELINE = "baseline"
    MODIFIED_LOSS = "modified_loss"
    SEMANTIC = "semantic"
    SEMANTIC_HIER = "semantic_hier"
    LAS_GLOBAL = "las_global"