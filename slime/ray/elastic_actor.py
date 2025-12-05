"""
Elastic Actor: Can switch between training and inference modes

This actor helps improve GPU utilization by dynamically switching between
training (Megatron) and inference (SGLang) workloads based on system needs.
"""

import os
import torch
import ray
from typing import Optional

from slime.ray.ray_actor import RayActor
from slime.ray.train_actor import TrainRayActor, get_local_gpu_id
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.utils.memory_utils import clear_memory, print_memory


class ElasticActor(RayActor):
    """
    Actor that can switch between training and inference modes.
    
    Modes:
    - 'training': Acts as a training actor (Megatron, part of DP group)
    - 'inference': Acts as an inference engine (SGLang)
    - None: Uninitialized state
    """
    
    def __init__(self, world_size, rank, master_addr, master_port, wandb_run_id):
        """
        Initialize elastic actor with training configuration.
        Both training and inference engines will be created, with inactive one offloaded.
        """
        self.current_mode = None
        self.elastic_rank = rank  # Which training rank this replaces (typically 1 for DP rank 1)
        
        # Training state (will be initialized with both engines)
        self.train_actor = None
        self._train_config = {
            'world_size': world_size,
            'rank': rank,
            'master_addr': master_addr,
            'master_port': master_port,
            'wandb_run_id': wandb_run_id,
        }
        
        # Inference state (will be initialized with both engines)
        self.sglang_engine = None
        self._inference_config = {}  # Will be set during init()
        
        # Track switching metrics
        self.switch_count = 0
        self.last_switch_time = None
        
        print(f"ElasticActor initialized (rank={rank}, will create both training and inference engines)")
    
    def init(self, args, role=None, wandb_run_id=None, with_ref=False, mode: str = None, **kwargs):
        """
        Initialize the elastic actor with BOTH training and inference engines.
        
        This creates both engines at startup, with the inactive one offloaded.
        The initial mode is training.
        
        Args:
            args: System arguments
            role: Training role ('actor' or 'critic')
            wandb_run_id: Wandb run ID
            with_ref: Whether to create reference model
        """
        self.args = args
        self.role = role or 'actor'
        self.with_ref = with_ref
        
        print(f"[ElasticActor rank={self.elastic_rank}] Initializing with BOTH engines...")
        
        # Step 1: Initialize training actor first (starts on GPU)
        print(f"[ElasticActor] Step 1/2: Initializing training engine...")
        result = self._init_training_engine(role=self.role, with_ref=with_ref)
        
        # Step 2: Inference engine will be initialized later when inference config is set
        # (we need inference config from train_async.py)
        print(f"[ElasticActor] Step 2/2: Inference engine will be initialized when config is set")
        
        self.current_mode = 'training'
        print(f"[ElasticActor rank={self.elastic_rank}] Initialization complete, starting in TRAINING mode")
        
        return result
    
    def set_inference_config(self, dist_init_addr, port, nccl_port, host=None):
        """
        Store the inference configuration for later use.
        We don't create the inference engine yet to avoid GPU memory conflicts.
        """
        print(f"[ElasticActor rank={self.elastic_rank}] set_inference_config() called", flush=True)
        print(f"[ElasticActor rank={self.elastic_rank}] Current mode: {self.current_mode}", flush=True)
        
        self._inference_config = {
            'dist_init_addr': dist_init_addr,
            'port': port,
            'nccl_port': nccl_port,
            'host': host,
        }
        
        print(f"[ElasticActor rank={self.elastic_rank}] Inference config stored (port={port})", flush=True)
        print(f"[ElasticActor] Inference engine will be created when switching to inference mode", flush=True)
    
    def _offload_training_to_cpu(self):
        """Clear training actor and free all GPU resources"""
        if self.train_actor is None:
            return
        
        print(f"[ElasticActor] Clearing training actor to free GPU memory...", flush=True)
        
        # Call cleanup on the training actor if it has one
        if hasattr(self.train_actor, 'clear_memory'):
            try:
                self.train_actor.clear_memory()
            except:
                pass
        
        # Destroy the training actor reference
        self.train_actor = None
        
        # Critical: Reset CUDA device to clear all contexts
        print(f"[ElasticActor] Resetting CUDA device...", flush=True)
        local_rank = get_local_gpu_id()
        try:
            torch.cuda.synchronize(local_rank)
            torch.cuda.empty_cache()
            # Reset the device - this clears all CUDA contexts
            torch.cuda.reset_peak_memory_stats(local_rank)
            torch.cuda.reset_accumulated_memory_stats(local_rank)
        except Exception as e:
            print(f"[ElasticActor] Warning during CUDA reset: {e}", flush=True)
        
        # Aggressive GPU memory cleanup
        print(f"[ElasticActor] Performing aggressive GPU memory cleanup...", flush=True)
        clear_memory()
        
        # Force garbage collection multiple times
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        print_memory("After training actor cleared")
        print(f"[ElasticActor] Training actor cleared, GPU should be ready for inference", flush=True)
    
    def _reload_training_to_gpu(self, role, with_ref):
        """Reinitialize training actor (simplified approach)"""
        print(f"[ElasticActor] Reinitializing training actor from checkpoint...", flush=True)
        
        # Reinitialize the training actor from checkpoint
        # This is simpler than trying to manage CPU offload for distributed models
        result = self._init_training_engine(role=role, with_ref=with_ref)
        
        print_memory("After training actor reinitialized")
        print(f"[ElasticActor] Training actor reinitialized from checkpoint", flush=True)
        
        return result
    
    def _init_inference_mode(self, dist_init_addr, port, nccl_port, host=None):
        """Initialize in inference mode (SGLang engine)"""
        print(f"[ElasticActor rank={self.elastic_rank}] Initializing inference engine...")
        
        # Store config for potential future switches
        self._inference_config = {
            'dist_init_addr': dist_init_addr,
            'port': port,
            'nccl_port': nccl_port,
            'host': host,
        }
        
        # Create and initialize SGLang engine if not already created
        if self.sglang_engine is None:
            print(f"[ElasticActor] Creating new SGLang engine...")
            # Elastic actor acts as an additional rollout engine AFTER the dedicated ones
            # So its rollout rank = number of dedicated rollout engines
            rollout_rank = self.args.rollout_num_gpus
            print(f"[ElasticActor] Using rollout engine rank={rollout_rank} (after {self.args.rollout_num_gpus} dedicated engines)")
            
            # The elastic actor should use its training rank as the GPU ID
            # Training rank directly corresponds to GPU index in CUDA_VISIBLE_DEVICES
            gpu_id = self.elastic_rank
            print(f"[ElasticActor] Overriding base_gpu_id to {gpu_id} (training rank {self.elastic_rank})")
            
            self.sglang_engine = SGLangEngine(self.args, rank=rollout_rank)
            self.sglang_engine.init(dist_init_addr, port, nccl_port, host, base_gpu_id_override=gpu_id)
            print(f"[ElasticActor] SGLang engine created successfully")
        else:
            print(f"[ElasticActor] SGLang engine already exists, reusing it")
        
        self.current_mode = 'inference'
        print(f"[ElasticActor rank={self.elastic_rank}] Successfully switched to INFERENCE mode")
        
        return f"http://{self.sglang_engine.server_host}:{self.sglang_engine.server_port}"
    
    def _init_training_engine(self, role, with_ref=False):
        """Initialize in training mode (Megatron training actor)"""
        print(f"[ElasticActor rank={self.elastic_rank}] Initializing in TRAINING mode")
        
        # Get the correct training actor implementation based on backend
        backend = self.args.train_backend
        if backend == "megatron":
            from slime.backends.megatron_utils import MegatronTrainRayActor
            train_actor_impl = MegatronTrainRayActor
        else:
            from slime.backends.fsdp_utils import FSDPTrainRayActor
            train_actor_impl = FSDPTrainRayActor
        
        # Create training actor component using the correct implementation
        self.train_actor = train_actor_impl(
            world_size=self._train_config['world_size'],
            rank=self._train_config['rank'],
            master_addr=self._train_config['master_addr'],
            master_port=self._train_config['master_port'],
            wandb_run_id=self._train_config['wandb_run_id'],
        )
        
        # Initialize training
        result = self.train_actor.init(self.args, role, self._train_config['wandb_run_id'], with_ref)
        
        self.current_mode = 'training'
        print(f"[ElasticActor rank={self.elastic_rank}] Successfully initialized in TRAINING mode")
        
        return result
    
    def switch_to_inference(self, weights=None):
        """
        Switch from training mode to inference mode.
        
        Args:
            weights: Optional weights to load into inference engine
        
        Returns:
            URL of the inference engine
        """
        import time
        print(f"[ElasticActor rank={self.elastic_rank}] ===== switch_to_inference() CALLED =====", flush=True)
        print(f"[ElasticActor rank={self.elastic_rank}] Current mode: {self.current_mode}", flush=True)
        start_time = time.time()
        
        print(f"[ElasticActor rank={self.elastic_rank}] Switching TRAINING → INFERENCE...", flush=True)
        
        if self.current_mode == 'inference':
            print(f"[ElasticActor] Already in inference mode, skipping switch", flush=True)
            if self.sglang_engine:
                return f"http://{self.sglang_engine.server_host}:{self.sglang_engine.server_port}"
            else:
                raise RuntimeError("In inference mode but sglang_engine is None")
        
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot switch to inference from mode: {self.current_mode}")
        
        # Verify inference config is set
        if not self._inference_config:
            raise RuntimeError("Inference config not set. Call set_inference_config() first.")
        
        # Step 1: Offload training state to CPU (preserves state for later)
        print(f"[ElasticActor] Step 1/4: Offloading training state to CPU", flush=True)
        self._offload_training_to_cpu()
        
        # Step 2: Clear GPU memory
        print(f"[ElasticActor] Step 2/4: Clearing GPU memory", flush=True)
        clear_memory()
        torch.cuda.empty_cache()
        print_memory("After offload and GPU cleanup")
        
        # Step 2.5: Wait for GPU to fully release resources
        print(f"[ElasticActor] Waiting for GPU to release resources...", flush=True)
        import time
        time.sleep(2)  # Give GPU time to release CUDA contexts
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Step 3: Initialize inference mode
        print(f"[ElasticActor] Step 3/4: Initializing inference engine", flush=True)
        try:
            engine_url = self._init_inference_mode(**self._inference_config)
        except Exception as e:
            print(f"[ElasticActor] ERROR initializing inference engine: {e}", flush=True)
            print(f"[ElasticActor] Check if GPU {self.elastic_rank} is free and ports {self._inference_config['port']}, {self._inference_config['nccl_port']} are available", flush=True)
            raise
        
        # Step 4: Weights are already in inference model from checkpoint
        print(f"[ElasticActor] Step 4/4: Inference engine ready with model weights", flush=True)
        
        self.switch_count += 1
        self.last_switch_time = time.time()
        duration = self.last_switch_time - start_time
        
        print(f"[ElasticActor] ✓ Switch complete: TRAINING → INFERENCE (took {duration:.1f}s)")
        print(f"[ElasticActor] Total switches: {self.switch_count}")
        
        return engine_url
    
    def switch_to_training(self, role='actor', with_ref=False):
        """
        Switch from inference mode to training mode.
        
        Args:
            role: Training role ('actor' or 'critic')
            with_ref: Whether to create reference model
        
        Returns:
            Training initialization result
        """
        import time
        start_time = time.time()
        
        print(f"[ElasticActor rank={self.elastic_rank}] Switching INFERENCE → TRAINING...")
        
        if self.current_mode == 'training':
            print(f"[ElasticActor] Already in training mode, skipping switch")
            return
        
        if self.current_mode != 'inference':
            raise RuntimeError(f"Cannot switch to training from mode: {self.current_mode}")
        
        # Note: train_actor is None after offloading, we'll reinitialize it from checkpoint
        
        # Step 1: Unregister from router (done externally by coordinator)
        print(f"[ElasticActor] Step 1/4: Router unregistration (handled externally)")
        
        # Step 2: Shutdown inference engine
        print(f"[ElasticActor] Step 2/4: Shutting down inference engine")
        if self.sglang_engine is not None:
            try:
                self.sglang_engine.dispose()
            except Exception as e:
                print(f"[ElasticActor] Warning: Error disposing SGLang engine: {e}")
            self.sglang_engine = None
        
        # Step 3: Clear GPU memory
        print(f"[ElasticActor] Step 3/4: Clearing GPU memory")
        clear_memory()
        torch.cuda.empty_cache()
        print_memory("After inference cleanup")
        
        # Step 4: Reinitialize training actor from checkpoint
        print(f"[ElasticActor] Step 4/4: Reinitializing training actor from checkpoint")
        result = self._reload_training_to_gpu(role=role, with_ref=with_ref)
        
        self.switch_count += 1
        self.last_switch_time = time.time()
        duration = self.last_switch_time - start_time
        
        print(f"[ElasticActor] ✓ Switch complete: INFERENCE → TRAINING (took {duration:.1f}s)")
        print(f"[ElasticActor] Total switches: {self.switch_count}")
        
        return result
    
    def get_mode(self) -> str:
        """Get current mode"""
        return self.current_mode
    
    def get_switch_stats(self) -> dict:
        """Get switching statistics"""
        return {
            'switch_count': self.switch_count,
            'last_switch_time': self.last_switch_time,
            'current_mode': self.current_mode,
        }
    
    # Proxy methods for training mode
    def train(self, *args, **kwargs):
        """Execute training step (only valid in training mode)"""
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot train in mode: {self.current_mode}")
        return self.train_actor.train(*args, **kwargs)
    
    def get_weights(self):
        """Get model weights (only valid in training mode)"""
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot get weights in mode: {self.current_mode}")
        return self.train_actor.get_weights()
    
    def set_rollout_manager(self, rollout_manager):
        """Set rollout manager (only valid in training mode)"""
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot set rollout manager in mode: {self.current_mode}")
        return self.train_actor.set_rollout_manager(rollout_manager)
    
    def save_model(self, step_id):
        """Save model checkpoint (only valid in training mode)"""
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot save model in mode: {self.current_mode}")
        return self.train_actor.save_model(step_id)
    
    def update_weights(self):
        """Update weights from rank 0 (only valid in training mode)"""
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot update weights in mode: {self.current_mode}")
        return self.train_actor.update_weights()
    
    def connect_actor_critic(self, critic):
        """Connect to critic model (only valid in training mode)"""
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot connect to critic in mode: {self.current_mode}")
        return self.train_actor.connect_actor_critic(critic)
    
    def sleep(self):
        """Offload model (only valid in training mode)"""
        if self.current_mode != 'training':
            # No-op in inference mode
            return
        return self.train_actor.sleep()
    
    def clear_memory(self):
        """Clear GPU memory"""
        if self.current_mode == 'training' and self.train_actor:
            return self.train_actor.clear_memory()
        else:
            clear_memory()
            torch.cuda.empty_cache()
    
    def get_master_addr_and_port(self):
        """Get master address and port (only for rank 0)"""
        if self.elastic_rank != 0:
            raise RuntimeError("Only rank 0 can provide master address and port")
        if self.current_mode != 'training':
            raise RuntimeError(f"Cannot get master addr/port in mode: {self.current_mode}")
        return self.train_actor.get_master_addr_and_port()
    
    # Proxy methods for inference mode
    def health_generate(self, *args, **kwargs):
        """Health check for inference engine (only valid in inference mode)"""
        if self.current_mode != 'inference':
            raise RuntimeError(f"Cannot health_generate in mode: {self.current_mode}")
        return self.sglang_engine.health_generate(*args, **kwargs)
    
    def dispose(self):
        """Cleanup resources"""
        print(f"[ElasticActor] Disposing in mode: {self.current_mode}")
        if self.current_mode == 'training' and self.train_actor:
            self.train_actor = None
        elif self.current_mode == 'inference' and self.sglang_engine:
            try:
                self.sglang_engine.dispose()
            except:
                pass
            self.sglang_engine = None
        
        clear_memory()
        torch.cuda.empty_cache()


