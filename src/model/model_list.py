from .bert_ner import BERTNER

model_list = {
    'bert_ner': BERTNER
}


def get_model(config, is_training):
    assert config.current_model in model_list

    model = model_list[config.current_model](config, is_training=is_training)

    return model
