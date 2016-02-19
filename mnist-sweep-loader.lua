require 'torch'
require 'io'

local MnistSweepLoader = {}
MnistSweepLoader.__index = MnistSweepLoader

function MnistSweepLoader.create(batch_size, use_cuda)
    local self = {}
    setmetatable(self, MnistSweepLoader)
    self.batch_size = batch_size

    self.data_split = {}
    self.data_split[1] = torch.load('mnist.t7/train_32x32.t7', 'ascii')
    self.data_split[2] = torch.load('mnist.t7/test_32x32.t7', 'ascii')

    self.data_split[1].data, mean, std = MnistSweepLoader.normalizeGlobal(
                                              self.data_split[1].data:float())
    self.data_split[2].data, _, _ = MnistSweepLoader.normalizeGlobal(
                                                self.data_split[2].data:float(),
                                                mean, std)

    self.num_batches = {}
    for i = 1, 2 do
        local randperm = torch.randperm(self.data_split[i].data:size(1)):long()
        self.data_split[i].data = self.data_split[i].data:index(1, randperm):squeeze()
        self.data_split[i].labels = self.data_split[i].labels:index(1, randperm)
        if use_cuda then
            self.data_split[i].data = self.data_split[i].data:float():cuda()
            self.data_split[i].labels = self.data_split[i].labels:float():cuda()
        end
        self.num_batches[i] = math.floor(self.data_split[i].data:size(1) / self.batch_size)
    end
    self.batch_idx = {0, 0}
    return self
end


function MnistSweepLoader:next_batch(split_id)
    self.batch_idx[split_id] = self.batch_idx[split_id] % self.num_batches[split_id] + 1
    local idx1 = (self.batch_idx[split_id] - 1) * self.batch_size + 1
    local idx2 = self.batch_idx[split_id] * self.batch_size 
    local x = self.data_split[split_id].data:sub(idx1, idx2)
    local y = self.data_split[split_id].labels:sub(idx1, idx2)
    return x, y
end


function MnistSweepLoader:reset_batch_pointer(split_id)
    self.batch_idx[split_id] = 0
end


function MnistSweepLoader.normalizeGlobal(data, mean_, std_)
    local std = std_ or data:float():std()
    local mean = mean_ or data:float():mean()
    data:add(-mean)
    data:mul(1/std)
    return data, mean, std
end 

return MnistSweepLoader
