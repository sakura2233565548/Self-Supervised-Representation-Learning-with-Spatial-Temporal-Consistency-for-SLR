def get_pretraining_set(opts):

    from feeder.feeder_pretraining import Feeder_SLR
    training_data = Feeder_SLR(**opts.train_feeder_args)

    return training_data


def get_finetune_training_set(opts):

    from feeder.feeder_downstream import Feeder_SLR

    data = Feeder_SLR(**opts.train_feeder_args)

    return data

def get_finetune_validation_set(opts):

    from feeder.feeder_downstream import Feeder_SLR
    data = Feeder_SLR(**opts.test_feeder_args)

    return data
