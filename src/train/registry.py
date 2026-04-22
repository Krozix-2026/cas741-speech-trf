from config.enums import Dataset, TaskName, LabelType

from train.trainer_librispeech import run_once as run_librispeech_lstm
from train.trainer_librispeech_phonemes_srv import run_once as run_librispeech_phone_srv
from train.trainer_librispeech_words import run_once as run_librispeech_word_onehot
from train.trainer_librispeech_words_srv import run_once as run_librispeech_word_srv
from train.trainer_librispeech_words_srv_semantic import run_once as run_librispeech_word_sem
from train.trainer_librispeech_mocha_words import run_once as run_las_mocha_words
from train.trainer_librispeech_las_words import run_once as run_las_words
from train.trainer_rnnt import run_once as run_rnnt
from train.trainer_burgundy import run_once as run_burgundy


TRAINER_REGISTRY = {
    (Dataset.LIBRISPEECH, TaskName.LSTM, None): run_librispeech_lstm,
    (Dataset.LIBRISPEECH, TaskName.LSTM_PHONE, LabelType.SRV): run_librispeech_phone_srv,
    (Dataset.LIBRISPEECH, TaskName.LSTM_WORD, LabelType.ONEHOT): run_librispeech_word_onehot,
    (Dataset.LIBRISPEECH, TaskName.LSTM_WORD, LabelType.SRV): run_librispeech_word_srv,
    (Dataset.LIBRISPEECH, TaskName.LSTM_WORD_SEM, LabelType.SRV): run_librispeech_word_sem,
    (Dataset.LIBRISPEECH, TaskName.LAS_MOCHA_WORDS, None): run_las_mocha_words,
    (Dataset.LIBRISPEECH, TaskName.LAS_WORDS, None): run_las_words,
    (Dataset.LIBRISPEECH, TaskName.RNNT, None): run_rnnt,
    (Dataset.BURGUNDY, TaskName.LSTM, None): run_burgundy,
}


def get_trainer(cfg):
    key = (cfg.dataset, cfg.task_name, cfg.label_type)
    if key not in TRAINER_REGISTRY:
        raise KeyError(f"No trainer registered for {key}")
    return TRAINER_REGISTRY[key]