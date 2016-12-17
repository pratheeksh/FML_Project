local nn = require 'nn'
local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local Relu = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout
local model  = nn.Sequential()
 classes = {'cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi'}
model:add(nn.Reshape(1,20,130))
model:add(Convolution(1, 16, 5, 1))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(Convolution(16, 128, 5, 1))
model:add(Dropout(0.5))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(View(1*128*5*29))
model:add(Dropout(0.5))
model:add(Linear(1*128*5*29, 1600))
model:add(Tanh())
model:add(Dropout(0.5))
model:add(Linear(1600, 64))
model:add(Tanh())
model:add(Dropout(0.5))
model:add(Linear(64, #classes))
model:add(nn.LogSoftMax())
-- model:cuda()
input = torch.Tensor(1,20,130)
out = model:forward(input)
print(out:size())

return model
