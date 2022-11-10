# Based on PyTorch tutorial: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
# Alternatively, see for layer-wise profiling: https://github.com/kshitij12345/torchnnprofiler

import torch
import torchvision.models as models

from torch.profiler import profile, record_function, schedule, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):  # context manager to name code ranges with user specified names
        model(inputs)

# Self CPU excludes calls to children operators
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
# display separately per input shape
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=10))

# profiling memory
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# profile stack traces
with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    model(inputs)

# profiling long jobs

my_schedule = schedule(
    skip_first=10,  # excluded
    wait=5,  # each cycle starts with this wait
    warmup=1,  # discarded due to additional overhead
    active=3,  # profiled steps
    repeat=2)  # by default keeps repeating
