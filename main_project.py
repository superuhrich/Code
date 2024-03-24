import torch
import logging
import csv
from Models.ResNextModel import ResNextModel
from Models.SwinV2Model import SwinV2Model

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

file_handler = logging.FileHandler(f'/home/paul.uhrich/Project/Logs/Run{training_run_id}/runLog.log')
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

logger.info(device)

batch_sizes = [16, 32, 64, 128]
learning_rates = [0.01, 0.001, 0.0001]

models_to_test = [
    ResNextModel(project_directory, grain_type, device, logger),
    SwinV2Model(project_directory, grain_type, device, logger)
]

for lr in learning_rates:
    for bs in batch_sizes:
        for model in models_to_test:
            best_model = model.train_model(30)
            model.test_model(best_model)


