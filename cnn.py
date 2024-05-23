import torch
from torch import nn

class SimpleMiniCNN(nn.Modele):
  def __init__(self):
    super(SimpleMiniCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=0)
    self.relu = nn.ReLU()
    def forward(self, x):
      print ('Input:', x):
      x = self.conv1(x)
      print('After Convolution:', x)
      x =  self.relu(x)
      print('After ReLU:', x)
      return x.view(-1, 1*1*1)

model = SimpleMiniCNN()

with torch.nograd():
  new_weights =  torch.tensor([[[[0.1, -0.2, 0.3, 0.4]]]])
  new_bias =  torch.tensor([-0.1])
  model.conv1.weight = nn.Parameter(new_weights)
  model.conv1.bias = nn.Parameter(new_bias)


input_tensor = torch.tensoe([[[[1.0, 2.0, 3.0, 4.0]]]])
output = model(input_tensor)
print ('Final Output', output)


##target for example only
target = torch.tensor([3.0])



loss_fn = nn.MSELoss()
loss = loss_fn(output, target)
print ('Loss', loss.item())

### clear old gradients

model.zero_grad()
loss.backward()

### learning rate
lr = 0.01

with torch.no_grad():
  for param in model.parameters():
    param -= lr* param.grad
    print(param)
