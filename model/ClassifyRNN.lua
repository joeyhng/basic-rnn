require 'model.RNNseq'
ClassifyRNN = torch.class('ClassifyRNN', 'RNNseq')

function ClassifyRNN:create_classifier(rnn_size, opt)
  print('children create classifier: ClassifyRNN')
  assert(opt.task == 'classify')
  assert(opt.loss_func == 'softmax' or opt.loss_func == 'logistic')
  self.dense_predict = opt.dense_predict or false

  self.classifier = nn.Sequential()
  if self.dense_predict then
    self.classifier:add(nn.JoinTable(1))
  end

  self.classifier:add(nn.Linear(rnn_size, rnn_size))
  self.classifier:add(nn.ReLU())
  self.classifier:add(nn.Linear(rnn_size, opt.num_targets))
  if opt.loss_func == 'softmax' then
    self.classifier:add(nn.LogSoftMax())
    self.criterion = nn.ClassNLLCriterion()
  elseif opt.loss_func == 'logistic' then
    self.classifier:add(nn.Sigmoid())
    self.criterion = nn.BCECriterion()
  end

end

----------------------- Forward and Backward for "find" task ------------------
-- criterion forward/backward passing length for normalizing various sequence length
function ClassifyRNN:forward(x, y)
  local len = x:size(2)
  self.rnn_state = self:rnn_forward(x)

  if self.dense_predict then
    self.label_transformer = nn.Sequential()
    self.label_transformer:add(nn.Replicate(len, 1, 1))
    self.label_transformer:add(nn.Reshape(x:size(1) * x:size(2), -1, false))
    if self.gpuid >= 0 then self.label_transformer:cuda() end

    if y then y = self.label_transformer:forward(y) end

    self.state_table = {}
    for i=1,len do
      self.state_table[i] = self.rnn_state[i][self.num_layers]
    end
    self.predictions = self.classifier:forward(self.state_table)
  else
    self.label_transformer = nn.Identity()
    self.predictions = self.classifier:forward(self.rnn_state[len][self.num_layers])
  end

  if y then
    return self.criterion:forward(self.predictions, y, len) 
  end
  return self.predictions
end

function ClassifyRNN:backward(x, y, classifier_only)
  local len = x:size(2)
  y = self.label_transformer:forward(y)
  local doutput = self.criterion:backward(self.predictions, y, len)
  local dstate
  if self.dense_predict then
    dstate = self.classifier:backward(self.state_table, doutput)
  else
    dstate = self.classifier:backward(self.rnn_state[len][self.num_layers], doutput)
  end
  if not classifier_only then
    self:rnn_backward(x, dstate)
  end
end

return ClassifyRNN
