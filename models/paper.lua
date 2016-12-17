local nn = require 'nn'
local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local Relu = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout
local model = nn.Sequential()
local SBatchNorm = nn.SpatialBatchNormalization

classes = { 'cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi' }
model:add(nn.Reshape(1, 20, 130))
model:add(Convolution(1, 32, 3, 3, 1, 1, 1, 1))
model:add(SBatchNorm(32))
model:add(Relu())
model:add(Max(2, 2, 2, 2))
model:add(Convolution(32, 64, 3, 3, 1, 1, 1, 1))
model:add(SBatchNorm(64))

model:add(Dropout(0.5))
model:add(Relu())
model:add(Max(2, 2, 2, 2))
model:add(Convolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(SBatchNorm(128))

model:add(Dropout(0.5))
model:add(Relu())
model:add(Max(2, 2, 2, 2))
model:add(Convolution(128, 256, 3, 3, 1, 1, 1, 1))
model:add(SBatchNorm(256))

model:add(Dropout(0.5))
model:add(Relu())
model:add(Max(2, 2, 2, 2))

model:add(View(1 * 256 * 1 * 8))
model:add(Dropout(0.5))
model:add(Linear(1 * 256 * 1 * 8, 1600))
model:add(Relu())
model:add(Dropout(0.5))
model:add(Linear(1600, 64))
model:add(Relu())
model:add(Dropout(0.5))
model:add(Linear(64, #classes))
model:add(nn.LogSoftMax())
-- model:cuda()
input = torch.Tensor(1, 20, 130)
out = model:forward(input)
print(out:size())

return model
