import torch.nn as nn 
import torch
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.loader = self.bias_loader
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor):
        param.data.copy_(loaded_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def _load_sharded_dim0(self, param: nn.Parameter, loaded_tensor: torch.Tensor):
        param_data = param.data
        # full_dim on the output column
        full_size = loaded_tensor.size(0)
        # dim size after sharding
        shard_size = full_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        sliced_tensor = loaded_tensor.narrow(0, start_index, shard_size)
        param_data.copy_(sliced_tensor)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # weight: [out_features, in_features]
        # Column parallel shards weight along output dimension.
        self._load_sharded_dim0(param, loaded_weights)

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor):
        # bias: [out_features]
        # Column parallel shards bias along output dimension as well.
        self._load_sharded_dim0(param, loaded_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# an extension of ColumnParallelLinear by merging several matrices
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def _load_merged_sharded_dim0(self, param: nn.Parameter, loaded_tensor: torch.Tensor, loaded_part_id: int):
        param_data = param.data
        # compute offset 
        offset = sum(self.output_sizes[:loaded_part_id]) // self.tp_size
        # compute size
        shard_size = self.output_sizes[loaded_part_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_start_index = self.tp_rank * shard_size
        shard_tensor = loaded_tensor.narrow(0, loaded_start_index, shard_size)
        param_data.copy_(shard_tensor)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        self._load_merged_sharded_dim0(param, loaded_weights, loaded_weight_id)

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor, loaded_bias_id: int):
        self._load_merged_sharded_dim0(param, loaded_bias, loaded_bias_id)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Calculate per-GPU output size
        # self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    def _load_qkv_sharded_dim0(self, param: nn.Parameter, loaded_tensor: torch.Tensor, load_weight_id: str):
        # batch_size * num_heads * num_token * head_size
        param_data = param.data
        # compute offset and shard size
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == "v":
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")
        
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_start_index = self.tp_rank * shard_size
        shard_weights = loaded_tensor.narrow(0, loaded_start_index, shard_size)

        param_data.copy_(shard_weights)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        self._load_qkv_sharded_dim0(param, loaded_weights, load_weight_id)
       
    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor, load_bias_id: str):
        self._load_qkv_sharded_dim0(param, loaded_bias, load_bias_id)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # full_dim on the input row
        full_data_input_size = loaded_weights.size(1)
        # dim size after sharding
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor):
        # bias: [out_features]
        # Row parallel does not shard bias.
        # Each rank keeps a full copy of the bias.
        param.data.copy_(loaded_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do not add bias before all_reduce.
        # Otherwise, each rank adds bias once and all_reduce will sum them.
        result = nn.functional.linear(x, self.weight, None)

        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)

        if self.bias is not None:
            result = result + self.bias

        return result


if __name__ == "__main__":
    # Example usage
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)