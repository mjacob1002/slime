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
        Note: Starts uninitialized, call init() to set initial mode.
        """
        self.current_mode = None
        self.elastic_rank = rank  # Which training rank this replaces (typically 1 for DP rank 1)
        
        # Training state (initialized when in training mode)
        self.train_actor = None
        self._train_config = {
            'world_size': world_size,
            'rank': rank,
            'master_addr': master_addr,
            'master_port': master_port,
            'wandb_run_id': wandb_run_id,
        }
        
        # Inference state (initialized when in inference mode)
        self.sglang_engine = None
        self._inference_config = {}  # Will be set during init()
        
        # Track switching metrics
        self.switch_count = 0
        self.last_switch_time = None
        
        print(f"ElasticActor initialized (rank={rank}, uninitialized mode)")
    
    def init(self, args, role=None, wandb_run_id=None, with_ref=False, mode: str = None, **kwargs):
        """
        Initialize the elastic actor.
        
        This method has two calling conventions:
        1. Training-style (called by RayTrainGroup.async_init):
           init(args, role, wandb_run_id, with_ref=False)
        2. Inference-style (called manually for inference initialization):
           init(args, mode='inference', dist_init_addr=..., port=..., nccl_port=..., host=...)
        
        Args:
            args: System arguments
            role: Training role ('actor' or 'critic') - for training mode
            wandb_run_id: Wandb run ID - for training mode
            with_ref: Whether to create reference model - for training mode
            mode: Explicit mode ('training' or 'inference')
            **kwargs: Additional mode-specific arguments for inference mode
        """
        self.args = args
        
        # Determine which calling convention is being used
        if mode == 'inference':
            # Inference-style initialization
            return self._init_inference_mode(**kwargs)
        elif mode == 'training' or role is not None:
            # Training-style initialization
            return self._init_training_mode(role=role or 'actor', with_ref=with_ref)
        else:
            # Default to training mode to maintain compatibility with RayTrainGroup
            print(f"[ElasticActor] No mode specified, defaulting to TRAINING mode")
            return self._init_training_mode(role='actor', with_ref=with_ref)
    
    def set_inference_config(self, dist_init_addr, port, nccl_port, host=None):
        """
        Set the inference configuration without initializing inference mode.
        This is useful for preparing the actor to switch to inference mode later.
        """
        print(f"[ElasticActor rank={self.elastic_rank}] set_inference_config() called", flush=True)
        print(f"[ElasticActor rank={self.elastic_rank}] Current mode: {self.current_mode}", flush=True)
        self._inference_config = {
            'dist_init_addr': dist_init_addr,
            'port': port,
            'nccl_port': nccl_port,
            'host': host,
        }
        print(f"[ElasticActor rank={self.elastic_rank}] Inference config set (port={port})", flush=True)
    
    def _init_inference_mode(self, dist_init_addr, port, nccl_port, host=None):
        """Initialize in inference mode (SGLang engine)"""
        print(f"[ElasticActor rank={self.elastic_rank}] Initializing in INFERENCE mode")
        
        # Store config for potential future switches
        self._inference_config = {
            'dist_init_addr': dist_init_addr,
            'port': port,
            'nccl_port': nccl_port,
            'host': host,
        }
        
        # Create and initialize SGLang engine
        # Note: We use rank=self.elastic_rank to maintain consistency
        self.sglang_engine = SGLangEngine(self.args, rank=self.elastic_rank)
        self.sglang_engine.init(dist_init_addr, port, nccl_port, host)
        
        self.current_mode = 'inference'
        print(f"[ElasticActor rank={self.elastic_rank}] Successfully initialized in INFERENCE mode")
        
        return f"http://{self.sglang_engine.server_host}:{self.sglang_engine.server_port}"
    
    def _init_training_mode(self, role, with_ref=False):
        """Initialize in training mode (Megatron training actor)"""
        print(f"[ElasticActor rank={self.elastic_rank}] Initializing in TRAINING mode")
        
        # Create training actor component
        self.train_actor = TrainRayActor(
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
        
        # Step 1: Save training state if needed (for potential resume)
        print(f"[ElasticActor] Step 1/5: Cleanup training state", flush=True)
        # Note: Training checkpoint is handled by TrainGroup, not individual actors
        
        # Step 2: Unload training model and optimizer
        print(f"[ElasticActor] Step 2/5: Unloading training components", flush=True)
        if self.train_actor is not None:
            # Clear references
            self.train_actor = None
        
        # Step 3: Clear GPU memory
        print(f"[ElasticActor] Step 3/5: Clearing GPU memory", flush=True)
        clear_memory()
        torch.cuda.empty_cache()
        print_memory(prefix="After training cleanup")
        
        # Step 4: Initialize inference mode
        print(f"[ElasticActor] Step 4/5: Initializing inference engine", flush=True)
        engine_url = self._init_inference_mode(**self._inference_config)
        
        # Step 5: Load weights if provided
        if weights is not None:
            print(f"[ElasticActor] Step 5/5: Loading weights into inference engine")
            # Weights are loaded via update_weights mechanism
            # SGLang will receive weights through the standard weight update flow
        else:
            print(f"[ElasticActor] Step 5/5: No weights to load (using existing model weights)")
        
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
        
        # Step 1: Unregister from router (done externally by coordinator)
        print(f"[ElasticActor] Step 1/5: Router unregistration (handled externally)")
        
        # Step 2: Wait for pending inference requests (if any)
        print(f"[ElasticActor] Step 2/5: Checking for pending requests")
        # SGLang automatically drains requests on shutdown
        
        # Step 3: Shutdown inference engine
        print(f"[ElasticActor] Step 3/5: Shutting down inference engine")
        if self.sglang_engine is not None:
            try:
                self.sglang_engine.dispose()
            except Exception as e:
                print(f"[ElasticActor] Warning: Error disposing SGLang engine: {e}")
            self.sglang_engine = None
        
        # Step 4: Clear GPU memory
        print(f"[ElasticActor] Step 4/5: Clearing GPU memory")
        clear_memory()
        torch.cuda.empty_cache()
        print_memory(prefix="After inference cleanup")
        
        # Step 5: Initialize training mode
        print(f"[ElasticActor] Step 5/5: Initializing training components")
        result = self._init_training_mode(role=role, with_ref=with_ref)
        
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

