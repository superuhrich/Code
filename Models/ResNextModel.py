from Models.ModelBase import BaseModelInterface
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from LearningAndTesting import TrainEvalTest

class ResNextModel(BaseModelInterface):
  def __init__(self, path, crop, device, logger, learning_rate, batch_size):
    super().__init__(path, crop, batch_size)
    self.device = device
    self.logger = logger

    model = torchvision.models.resnext50_32x4d(weights='IMAGENET1K_V1')
    #freeze convolutional layers
    for param in model.parameters():
      param.requires_grad = False
    
    # Customize the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model = model.to(self.device)

    # Define model parameters
    criterion = nn.CrossEntropyLoss()
    # only look at new top parameters (fully connected)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    # decrease learning reat by 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    self.model = model

    data_loaders, dataset_sizes = self.get_data()

    self.trainer_evaluator = TrainEvalTest(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, self.device, self.logger, self.__class__.__name__, learning_rate, batch_size)

  
  def train_model(self, epochs):
    self.trainer_evaluator.train_model(epochs)

  def test_model(self):
    self.trainer_evaluator.test_model()

    
