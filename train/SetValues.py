class SV(object):
    # Basic value set.
    mode="training"
    #sk_index=[[0],[1,5,9,13,17],[2,6,10,14,18],[3,7,11,15,19],[4,8,12,16,20]]
    sk_index=[0,1,6,11,16,2,7,12,17,3,8,13,18,4,9,14,19,5,10,15,20]
    input_size = 224
    heatmap_size = 28
    stages = 3
    joint = 21
    dataset_main_path = "dataset"
    #dataset_main_path = "../../../dataset/FreiHAND_pub_v2"
    model_save_path = "Model"
    model_name="cpm_hand"
    pretrained_model_name="cpm_hand"
    testdir="test"
    testname="00000001.jpg"

    #Training value set
    batch_size=10
    episodes=30
    epo_turns=5000
    learning_rate=0.0001
    lr_decay_rate = 0.6
    lr_decay_step=5000
