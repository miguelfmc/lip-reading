#Config file
model = dict (
	name = 'Model ',
	file = 'ble ble',
	check_dir = "/home/alemosan/lipReading/checkpoints/",
	mode = "Train",
	loading = False,
	path_load = "/home/alemosan/lipReading/checkpoints/"
)

parameters= dict(
	name = "SGD",
    lr = 0.01 ,
    lr_scheduler = "constant",
    lr_decay = 0.8,
    stage_length = 100,
    staircase = False,
    clip_norm = 1,
    momentum = 0.9
    )

data_args = dict (
	num_epochs = 5,
    batch_size = 32,
    train_size = 0.25 *500_000 #Number of training points
)


