class SV(object):
    #setting for dataset
    testdir = "test"
    testname = "hand.jpg"
    mode="training"
    dataset_main_path = "dataset"
    log_save_path = "log"
    #dataset_main_path = "../../../dataset/FreiHAND_pub_v2"


    #setting for the model
    stages=6
    joint = 21
    input_size = 368
    heatmap_size = input_size / 8
    model="cpm_sk"
    model_save_path = "Model"
    model_name = "sk_hand"
    pretrained_model_name ="sk_hand-60000"#"cpm_hand-225000"#"cpm_hand-225000"#"cpm_hand-90000" #"cpm_hand.pkl"#"sk_hand"
    sk_index = [0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20]

    #setting for training
    batch_size=5
    episodes=30
    epo_turns=5000
    learning_rate=0.00001
    lr_decay_rate = 0.95
    lr_decay_step=5000