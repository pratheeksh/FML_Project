require 'torch'
require 'cutorch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
matio = require 'matio'
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

require 'cunn'

local trainData = torch.load('train.t7')
-- local testData = torch.load(DATA_PATH..'test.t7')


local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 320, 140
local DATA_PATH = (opt.data ~= '' and opt.data or './data_/')

torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)

function resize(img)
    modimg = img[{ {}, { 200, 480 }, {} }]
    return image.scale(modimg, WIDTH, HEIGHT)
end

function yuv(img)
    return image.rgb2yuv(img)
end

function norm(img)
    new = img / 255
    new = new - torch.mean(new)
    return new
end

function transformInput(inp)
    f = tnt.transform.compose {
        [1] = resize,
        [2] = yuv,
        [3] = norm
    }
    return f(inp)
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
    print(label, filename, mattoload[filename])

    return mattoload[filename]
end

function getTrainLabel(dataset, idx)
    return dataset[idx][2]
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

function getIterator(dataset)

    return tnt.DatasetIterator {
        --	dataset =  tnt.ShuffleDataset{
        dataset = tnt.BatchDataset {
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end


trainDataset = tnt.SplitDataset {
    partitions = { train = 0.9, val = 0.1 },
    initialpartition = 'train',
    dataset = tnt.ShuffleDataset {
        dataset = tnt.ListDataset {
            list = torch.range(1, 6000):long(),
            load = function(idx)
                return {
                    input = getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}

--[[testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}
]]


local model = require("models/" .. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
--local criterion = nn.MSECriterion()--nn.CrossEntropyCriterion()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter { topk = { 1 } }
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
    -- clerr:add(state.network.output, state.sample.target)
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
        optimMethod = optim.adam,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            --[[momentum = opt.momentum,
		learningRateDecay = .01,
		weightDecay = .001--]]
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

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1


engine.hooks.onForward = function(state)
    local fileNames = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test {
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
