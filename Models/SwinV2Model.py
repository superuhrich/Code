from Models.ModelBase import BaseModelInterface
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from LearningAndTesting import TrainEvalTest


class SwinV2Model(BaseModelInterface):
  def __init__(self, device, logger, data_handler):
    self.device = device
    self.logger = logger
    self.data_handler = data_handler
    self.model = None
    self.trainer_evaluator = None
    
  def prepare_model(self, learning_rate, batch_size):

    model = torchvision.models.swin_v2_s(weights=torchvision.models.Swin_V2_S_Weights.DEFAULT)

    for param in model.parameters():
      param.requires_grad = False

    # Customize the fully connected layer
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 7)
    model = model.to(self.device)

    # Define model parameters
    criterion = nn.CrossEntropyLoss()
    # only look at new top parameters (fully connected)
    optimizer = optim.Adam(model.head.parameters(), lr=learning_rate)
    # decrease learning reat by 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    self.model = model

    data_loaders, dataset_sizes = self.data_handler.get_data(batch_size)

    self.trainer_evaluator = TrainEvalTest(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, self.device, self.logger, self.__class__.__name__, learning_rate, batch_size)  

  def train_model(self, epochs):
    self.trainer_evaluator.train_model(epochs)

  def test_model(self):
    self.trainer_evaluator.test_model()  

    

