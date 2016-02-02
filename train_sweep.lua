require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

local MnistSweepLoader = require('mnist-sweep-loader')
local RNNseq = require('model.RNNseq')
local ClassifyRNN = require('model.ClassifyRNN')

cmd = torch.CmdLine()
cmd:option('-model_type', 'gru', 'rnn or gru')
cmd:option('-gpuid', 0, 'gpuid')
cmd:option('-num_hidden', 128, 'rnn size')
cmd:option('-num_layers', 1, 'number of rnn layers')
cmd:option('-grad_clip', 5, 'gradient clipping value')
cmd:option('-batch_size', 64, 'batch size')
cmd:option('-learning_rate', 2e-3, 'batch size')
cmd:option('-max_epochs', 10, 'number of training epoch')
cmd:option('-seed',123,'torch manual random number generator seed')
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)


optim_state = {learningRate = opt.learning_rate, alpha = 0.95}

loader = MnistSweepLoader.create(opt.batch_size, opt.gpuid)

rnn_opt = {model_type = opt.model_type,
           rnn_size = opt.num_hidden,
           num_layers = opt.num_layers,
           input_size = 32,
           seq_length = 32}

classifier_opt = {task = 'classify',
                  loss_func = 'softmax',
                  num_targets = 10}
    
model = ClassifyRNN.new(rnn_opt, classifier_opt, opt.gpuid)

params, grad_params = model:getParameters()
params:uniform(-0.08, 0.08)


function feval(x)
  if x ~= params then params:copy(x) end
  grad_params:zero()
  local x, y = loader:next_batch(1)
  local loss = model:forward(x, y)
  model:backward(x, y)
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, grad_params
end


function eval_split(split_id)
  local n = loader.num_batches[split_id]
  loader:reset_batch_pointer(split_id)
  local confusion = optim.ConfusionMatrix(10)
  confusion:zero()
  for i = 1,n do
    local x, y = loader:next_batch(split_id)
    confusion:batchAdd(model:forward(x), y)
  end
  print(confusion)
  return confusion.totalValid
end


iterations = opt.max_epochs * loader.num_batches[1]
for i = 1,iterations do
  local epoch = i / loader.num_batches[1]
  local timer = torch.Timer()
  local _, loss = optim.rmsprop(feval, params, optim_state)
  loss = loss[1]
  local time = timer:time().real
  if i % 100 == 0 then
    print(string.format(
    '>> i=%d (epoch %.2f) train loss=%g,  step time = %.2fs',
    i, epoch, loss, time))
  end
  if i % 500 == 0 or i == iterations then
    print('---- iteration ' .. i ..' ----')
    eval_split(2)
    collectgarbage()
  end
end
