from DataHandling import DataHandler

class BaseModelInterface(DataHandler):
  def __init__(self, path, crop, batch_size):
    super().__init__(path,crop, batch_size)

  def train_model(self, epochs):
      raise NotImplementedError("Must implement train_model method")

  def test_model(self):
      raise NotImplementedError("Must implement test_model method")