import torch_pruning as tp
from archs import PMRIDu3
import torch
from utils.losses import PSNRLoss
model = PMRIDu3()

# for layer_name, param in model.named_parameters():
#     print(f"layer name: {layer_name} has {param.shape}")


# print(model.enc2.__dict__)
#tp.prune_conv_out_channels(model.enc2._modules['0'], idxs=[0,1]) # remove channel 0 and channel 1

DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))
example_inputs = torch.randn(1, 3, 256, 256)

# 1. Importance criterion
imp = tp.importance.GroupTaylorImportance() # or GroupNormImportance(p=2), GroupHessianImportance(), etc.

# 2. Initialize a pruner with the model and the importance criterion
# ignored_layers = []
# for m in model.modules():
#     if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
#         ignored_layers.append(m) # DO NOT prune the final classifier!

pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.9, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
    ignored_layers=[],
)
loss_f = PSNRLoss()
# 3. Prune & finetune the model
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
if isinstance(imp, tp.importance.GroupTaylorImportance):
    # Taylor expansion requires gradients for importance estimation
    loss_value = loss_f(model(example_inputs), example_inputs) # A dummy loss, please replace this line with your loss function and data!
    loss_value.backward()
pruner.step()


macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(base_macs, base_nparams)

print(macs, nparams )