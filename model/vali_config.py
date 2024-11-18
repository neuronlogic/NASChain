class ValidationConfig:
    def __init__(self):
        self.min_parameters = 0
        self.max_parameters = 0
        self.min_flops = 0
        self.max_flops = 0
        self.min_accuracy = 80.0
        self.max_accuracy = 100.0
        self.max_download_file_size = 5.5*1024*1024
        self.train_epochs = 50    
        self.max_flops = 280000000
        self.wandb_project = 'naschain-pareto'
        self.wandb_entitiy = 'naschain'
        self.learning_rate = 0.025