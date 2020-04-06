class SV(object):
    # Basic value set.
    mode="training"
    input_size = 224
    heatmap_size = 28
    cpm_stage = 3
    joint = 21
    #dataset_main_path = "dataset"
    dataset_main_path = "../../../dataset/FreiHAND_pub_v2"
    model_save_path = "Model"
    model_name="cpm_hand"
    pretrained_model_name="cpm_hand"

    #Training value set
    batch_size=5
    episodes=30
    epo_turns=1000
    learning_rate=0.001
    lr_decay_rate = 0.5
    lr_decay_step=3000
