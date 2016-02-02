local model_utils = require 'model_utils'
local RNN = require 'model.RNN'
local GRU = require 'model.GRU'

RNNseq = torch.class('RNNseq')

-- rnn_opt should be a table containing:
--    model_type: 'gru', or 'rnn'
--    input_size: input feature dimension
--    rnn_size:   # of cells in RNN
--    num_layers: # of layers for stacked RNN
--    seq_length: length of sequence
--    dropout:    dropout probability
--
-- classifier_opt should be a table containing:
--   task:         find, mem, classify
--   loss_func:    mse, softmax, logistic, or multiloss
--   num_targets:  number of targets to find
--   side_feature_dim:  side feature input dimension
--
function RNNseq:__init(rnn_opt, classifier_opt, gpuid)
  -- RNN Options
  print('RNN options: ')
  print(rnn_opt)
  self.gpuid = gpuid
  self.model_type = rnn_opt.model_type or 'gru'
  self.num_layers = rnn_opt.num_layers or 1
  self.rnn_size   = rnn_opt.rnn_size or 128
  self.input_size = rnn_opt.input_size
  self.seq_length = rnn_opt.seq_length 
  self.dropout = rnn_opt.dropout or 0

  self.protos = {}
  if self.model_type == 'gru' then
    self.protos.rnn = GRU.gru(self.input_size, self.rnn_size, self.num_layers, self.dropout)
  else
    self.protos.rnn = RNN.rnn(self.input_size, self.rnn_size, self.num_layers, self.dropout)
  end

  self:create_classifier(self.rnn_size, classifier_opt)

  if self.gpuid >= 0 then
    print('using GPU')
    for k,v in pairs(self.protos) do v:cuda() end
    self.classifier:cuda()
    self.criterion:cuda()
    if self.label_batch_concat then self.label_batch_concat:cuda() end
  end
  self:expand_rnn()
  return self
end

----------- function similar to torch nn module interface ------------
function RNNseq:getParameters()
  return self.params, self.grad_params
end

function RNNseq:evaluate()
  self.classifier:evaluate()
  for t = 1,#self.clones.rnn do
    self.clones.rnn[t]:evaluate()
  end
end

function RNNseq:training()
  self.classifier:training()
  for t = 1,#self.clones.rnn do
    self.clones.rnn[t]:training()
  end
end

---------------- Model util function -------------------------------
function RNNseq:expand_rnn()
  self.params, self.grad_params = model_utils.combine_all_parameters(
                                    self.protos.rnn,
                                    self.classifier)
  self.clones = {}
  for name, proto in pairs(self.protos) do
    self.clones[name] = model_utils.clone_many_times(proto, self.seq_length)
  end
end

function RNNseq:load_rnn_weights(checkpoint_path)
  local model = torch.load(checkpoint_path).model
  self.protos = model.protos
  self:expand_rnn()
end

function RNNseq.load_checkpoint(checkpoint_path)
  local self = torch.load(checkpoint_path).model
  self:expand_rnn()
  return self
end


function RNNseq:rnn_zero_states(batch_size)
  local zero_state = {}
  for L=1,self.num_layers do
    local z = torch.zeros(batch_size, self.rnn_size)
    if self.gpuid >= 0 then z = z:cuda() end
    table.insert(zero_state, z)
  end
  return zero_state
end

-------------- Logic for forward/backward pass for recurrent layers ------------
function RNNseq:rnn_forward(x)
  local len = x:size(2)
  self.rnn_state = {[0] = self:rnn_zero_states(x:size(1))}
  for t = 1,len do
    self.clones.rnn[t]:training()
    self.rnn_state[t] = self.clones.rnn[t]:forward{x[{{}, t}], unpack(self.rnn_state[t-1])}
    if type(self.rnn_state[t]) ~= 'table' then self.rnn_state[t] = {self.rnn_state[t]} end
  end
  return self.rnn_state
end

function RNNseq:rnn_backward(x, dstate)
  local len = x:size(2)
  local drnn_state = {[len] = self:rnn_zero_states(x:size(1))}
  if type(dstate) == 'table' then
    for i=1,len-1 do
      drnn_state[i] = {[self.num_layers] = dstate[i]}
    end
    drnn_state[len][self.num_layers] = dstate[len]
  else
    drnn_state[len][self.num_layers] = dstate
  end
  for t = len,1,-1 do
    if #drnn_state[t] == 1 then drnn_state[t] = drnn_state[t][1] end
    local dlst = self.clones.rnn[t]:backward({x[{{}, t, {}}], unpack(self.rnn_state[t-1])},
                                             drnn_state[t])
    -- update gradients
    drnn_state[t-1] = {}
    for i = 2,#dlst do
      if drnn_state[t-1][i-1] then
        drnn_state[t-1][i-1]:cadd(dlst[i])
      else
        drnn_state[t-1][i-1] = dlst[i]
      end
    end
  end
end

return RNNseq
