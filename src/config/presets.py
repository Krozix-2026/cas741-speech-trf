from config.schema import TrainConfig
from config.enums import Dataset, TaskName, LabelType, PolicyName


def librispeech_lstm(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM,
        policy_name=PolicyName.MODIFIED_LOSS,
        seed=seed,
    )


def librispeech_lstm_phone_srv(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM_PHONE,
        label_type=LabelType.SRV,
        policy_name=PolicyName.MODIFIED_LOSS,
        seed=seed,
        srv_dim=2048,
        srv_k=16,
        srv_loss="cosine",
        srv_value="pm1",
        srv_seed=123,
        phone_label_region="center",
        phone_center_ms=30,
    )


def librispeech_lstm_word_onehot(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM_WORD,
        label_type=LabelType.ONEHOT,
        policy_name=PolicyName.BASELINE,
        seed=seed,
    )


def librispeech_lstm_word_srv(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM_WORD,
        label_type=LabelType.SRV,
        policy_name=PolicyName.BASELINE,
        seed=seed,
        srv_dim=2048,
        srv_k=16,
        srv_loss="cosine",
        srv_value="pm1",
        srv_seed=123,
    )


def librispeech_lstm_word_semantic_hier(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM_WORD_SEM,
        label_type=LabelType.SRV,
        policy_name=PolicyName.SEMANTIC_HIER,
        seed=seed,
        srv_dim=2048,
        srv_k=16,
        srv_loss="cosine",
        srv_value="pm1",
        srv_seed=123,
        semantic_alpha=1.0,
        semantic_beta=0.5,
        word_rep_dim=256,
        word_lstm_hidden=256,
        word_lstm_layers=1,
        rep_dropout=0.1,
        word_dropout_p=0.25,
        semantic_shift=1,
        semantic_detach_frame=False,
    )


def librispeech_las_mocha_words(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LAS_MOCHA_WORDS,
        seed=seed,
        batch_size=128,
        epoch=200,
        lstm_layers=1,
        lstm_hidden=512,
        las_enc_subsample=2,
        las_max_words=200,
    )


def librispeech_las_words(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LAS_WORDS,
        policy_name=PolicyName.LAS_GLOBAL,
        seed=seed,
        batch_size=64,
        epoch=200,
        lstm_layers=2,
        lstm_hidden=256,
        bidirectional=True,
        las_enc_subsample=2,
        las_max_words=200,
    )


def burgundy_lstm(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        dataset=Dataset.BURGUNDY,
        task_name=TaskName.LSTM,
        policy_name=PolicyName.MODIFIED_LOSS,
        seed=seed,
    )


PRESETS = {
    "ls_lstm": librispeech_lstm,
    "ls_lstm_phone_srv": librispeech_lstm_phone_srv,
    "ls_lstm_word_onehot": librispeech_lstm_word_onehot,
    "ls_lstm_word_srv": librispeech_lstm_word_srv,
    "ls_lstm_word_semantic_hier": librispeech_lstm_word_semantic_hier,
    "ls_las_mocha_words": librispeech_las_mocha_words,
    "ls_las_words": librispeech_las_words,
    "burgundy_lstm": burgundy_lstm,
}


def get_preset(name: str, seed: int = 0) -> TrainConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name](seed=seed)