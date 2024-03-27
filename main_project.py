import torch
import logging
import csv
from Models.ResNextModel import ResNextModel
from Models.SwinV2Model import SwinV2Model
from Models.EfficienetV0 import EfficientNet0Model
from DataHandling import DataHandler

training_run_id = 1
project_directory ='/home/paul.uhrich/Project/Data'
grain_type = 'WR2021'

log_file_path = '/home/paul.uhrich/Project/Logs/trainValTest_log.csv'
confusion_matrix_path = '/home/paul.uhrich/Project/Logs/confusion_matrix_log.csv'

with open(log_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(['Model Name','Batch Size', 'Learning Rate', 'Phase', 'Epoch', 'Loss', 'Accuracy', 'Patience Count'])

with open(confusion_matrix_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(['Model Name','Batch Size', 'Learning Rate', 'Accuracy', 'Loss', 'Confusion Matrix'])

logger = logging.getLogger(f'Run{training_run_id}')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(f'/home/paul.uhrich/Project/Logs/runLog.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)





logger.info(f'Starting Transfer Learn for Run {training_run_id}')


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

logger.info(f'Using device {device}\n')

#We need a datahandler to be shared between all runs so that the data in train/eval/test is the same

data_handler = DataHandler(project_directory, grain_type)

batch_sizes = [16, 32, 64]
learning_rates = [0.1, 0.01]

models_to_test = [
    SwinV2Model(device, logger, data_handler),
    ResNextModel(device, logger, data_handler),
    EfficientNet0Model(device, logger, data_handler)
]

for model in models_to_test:
    for lr in learning_rates:
        for bs in batch_sizes:
            try:
                logger.info(f'Starting model train on {model.__class__.__name__} at learning Rate {lr} and batch size {bs}')
                model.prepare_model(lr, bs)
                best_model = model.train_model(30,5)
                model.test_model()
            except Exception as e:
                logger.error(f'An error occured within model {model.__class__.__name__} at Learning Rate {lr} and Batch size {bs}: Error {e}')
                continue


