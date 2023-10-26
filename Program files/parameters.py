# this is a class to contain all parameters for a run
# it also contains parameters generated at runtime


class RunParameters:

    # empirical hyperparameter 0.001; 1000; 10000; 512*4
    LEARNING_RATE = LR = 1 * 10 ** -3
    BATCH_SIZE = 1000
    MAX_EPOCHS = 10000
    FINGERPRINT_VECTOR_SIZE = FP_SIZE = 512 * 6  # only important for PFP creation
    EARLY_STOPPING_PATIENCE, MIN_DELTA = 20, 0.1  # early stopping settings
    # LAYER_DEPTHS = 3

    # choose map4_fp or morgan4_fp from csv with additional already created FPs or leave blank or pfp
    USE_FP = "pfp"
    pfp_const_type = None  # constitution type for pfp - either Subs+AP or morgan4 for the enhanced part
    model_nr = "N" + "_" + USE_FP  # name the model will be saved under

    export_path = None
