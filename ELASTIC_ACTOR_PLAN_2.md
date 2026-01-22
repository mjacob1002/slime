# Plan to implement Elastic Actors

Elastic actors are going to be used in conjunction with dedicated trainers and rollout instances. These elastic actors will allow a subset of the engines, in some class RayElasticActorGroup, to have switch_to_training and switch_to_inference modes. This document will list what we have tried prior, why it didn't work, and 
a plan for the next implementation (and how it should be done).

## Prior implementation
Previously, we tried to have each individual elastic actor own both a local training engine and inference engine. On weight updates, each local training engine would propogate an update to the local inference engine. During inference mode, the local inference engines would register with a router that was created, and then help with rollouts.
On training, the elastic actors would be onloaded and the inference engines would be offloaded. The trainers would then train. 

This approach suffered from issues when using the Megatron backend. Because the local engines were not Ray actors, there were numerous issues that occurred in reusing existing infrastructure for weight updates. We specifically ran into issues with the Torch Memory Saver, as offloading the megatron and sglang engines would result in interesting regions, throwing an error. This didn't happen in the normal colocated versions because they were separate processes. Ultimately, we want to enable to a subset of the actors (the ones that the user will say are elastic group) to behave like the engines do in the colocated version of RL training. We should be able to perform weight updates the same way, and the only difference is that this group is separate from the dedicated rollout and training engines. When doing RL training with elastic actors, the colocated flag, as well as the offload_train and offload_rollout flags in the training scripts, shouldn't be set, as the ElasticActors will implicitly do this offloading / onloading when doing switch_to_training or switch_to_inference. 

## API

The API shoud look something like this for the RayElasticActorGroup:

```python
class RayElasticActorGroup:

    def __init__(...):
        self.megatron_training_actors = ... # training actors
        self.sglang_inference_engines = ... # inference engines.
        # NOTE: self.sglang_inference_engines and self.megatron_training_actors should sit on the SAME GPUs. When you want to do training, you simply offload rollout and onload trainers. To do inference, you offload trainers and onload rollout. 
        # This should be same as the actors in the colocated setting. You should also propogate weight updates in the same way AMONG yourselves. In other words, the trainers that belong to the elastic group should be responsible for updating weights of the engines that belong to the elastic group
        self.mode = "inference" # can also be 'training'
    
    def switch_to_training(self):
        # Step 1: offload inference engines
        # Step 2: onload trainers
        self.mode = "training"
    
    def switch_to_inference(self):
        # Step 1: offload trainers engines
        # Step 2: onload inference 
        self.mode = "inference"    
    
    def generate(self):
        # generation code here

    def update_weights(self):
        # update the weights of te sglang_inference_engines
        pass
    
    # Should also include the rest of the API used by the training loops for inference and training API so that it can be used seamlessly by the user
```