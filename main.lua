require 'torch'
require 'cutorch'
optim = require 'optim'
require 'os'
require 'optim'
require 'xlua'
matio = require 'matio'
--local criterion =  nn.MultiLabelSoftMarginCriterion()
names = {}
matio.use_lua_strings = true
data = io.open("labels.csv", 'r')
cla = matio.load('../data/matfiles/cla.mat')
gac = matio.load('../data/matfiles/gac.mat')
org = matio.load('../data/matfiles/org.mat')
sax = matio.load('../data/matfiles/sax.mat')
vio = matio.load('../data/matfiles/vio.mat')
cel = matio.load('../data/matfiles/cel.mat')
flu = matio.load('../data/matfiles/flu.mat')
gel = matio.load('../data/matfiles/gel.mat')
pia = matio.load('../data/matfiles/pia.mat')
tru = matio.load('../data/matfiles/tru.mat')
voi = matio.load('../data/matfiles/voi.mat')

test =  matio.load('../data/matfiles/test.mat')
require 'cunn'

local trainData = torch.load('train.t7')
local testData = torch.load('test.t7')
local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 20, 130
local DATA_PATH = (opt.data ~= '' and opt.data or './data_/')

torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
function tablelength(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

function getTrainSample(dataset, idx)
    filename = dataset[idx][1]
    label = dataset[idx][2]
    --- 'cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi'
    if label == 'cla' then
        mattoload = cla
    elseif label == 'gac' then
        mattoload = gac
    elseif label == 'org' then
        mattoload = org
    elseif label == 'sax' then
        mattoload = sax
    elseif label == 'vio' then
        mattoload = vio
    elseif label == 'cel' then
        mattoload = cel
    elseif label == 'flu' then
        mattoload = flu
    elseif label == 'gel' then
        mattoload = gel
    elseif label == 'pia' then
        mattoload = pia
    elseif label == 'tru' then
        mattoload = tru
    else
        mattoload = voi
    end
    --   print(label, filename, mattoload[filename])
    return mattoload[filename]
end

function getTrainLabel(dataset, idx)
    label = dataset[idx][2]
    labelTensor = torch.Tensor(11):fill(0)
    if label == 'cla' then
        mattoload = 1
    elseif label == 'gac' then
        mattoload = 2
    elseif label == 'org' then
        mattoload = 3
    elseif label == 'sax' then
        mattoload = 4
    elseif label == 'vio' then
        mattoload = 5
    elseif label == 'cel' then
        mattoload = 6
    elseif label == 'flu' then
        mattoload = 7
    elseif label == 'gel' then
        mattoload = 8
    elseif label == 'pia' then
        mattoload = 9
    elseif label == 'tru' then
        mattoload = 10
    else
        mattoload = 11
    end
    labelTensor[{ mattoload }] = 1
    return torch.LongTensor { mattoload }
end

function getTestSample(dataset, idx)
  filename = dataset[idx][1]
  print(test[filename])
	return test[filename]
end

function getIterator(dataset)

    return tnt.DatasetIterator {
        dataset = tnt.BatchDataset {
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end

print(tablelength(trainData))
trainDataset = tnt.SplitDataset {
    partitions = { train = 0.9, val = 0.1 },
    initialpartition = 'train',
    dataset = tnt.ShuffleDataset {
        dataset = tnt.ListDataset {
            list = torch.range(1, tablelength(trainData)-500):long(),
            load = function(idx)
                return {
                    input = getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(501, tablelength(trainData)):long(),
    load = function(idx)
        return {
            input = getTestSample(trainData, idx),
            sampleId = getTrainLabel(trainData, idx)
        }
    end
}



local model = require("models/" .. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter { topk = { 1 } }
--local clerr = tnt.MultiLabelConfusionMeter{k=11}
local timer = tnt.TimeMeter()
local batch = 1
model:cuda()
criterion:cuda()

-- print(model)
engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

local trainAccuracy
local input = torch.CudaTensor()
local target = torch.CudaTensor()
engine.hooks.onSample = function(state)
    input:resize(state.sample.input:size()):copy(state.sample.input)
    state.sample.input = input
    if state.sample.target then
        target:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.target = target
    end
end


engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
            mode, batch, state.iterator.dataset:size(), meter:value())) -- , clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
        mode, meter:value(), clerr:value { k = 1 }, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train {
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum,
            learningRateDecay = .000001,
            weightDecay = .001
        }
    }

    trainDataset:select('val')
    engine:test {
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch ' .. tostring(epoch))
    epoch = epoch + 1
end
torch.save("model.t7", model:clearState())
local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1


engine.hooks.onForward = function(state)
--[[   local fileNames = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end]]--
   
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end
engine:test {
    network = model,
    iterator = getIterator(trainDataset)
}

print("The End!")
