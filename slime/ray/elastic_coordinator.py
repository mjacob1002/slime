"""
Elastic Actor Coordination Logic

This module handles the decision-making for when an elastic actor should switch
between training and inference modes based on queue state and work remaining.
"""

class ElasticSwitchingDecision:
    """Encapsulates the logic for deciding when elastic actor should switch modes"""
    
    # Configuration constants
    SWITCH_OVERHEAD_SECONDS = 15  # Time it takes to switch modes
    MIN_WORK_THRESHOLD = 20       # Only switch if at least this many groups remaining
    
    @classmethod
    def should_switch_to_inference(
        cls,
        progress: dict,
        current_throughput: float = 2.0,  # groups per second with single engine
    ) -> tuple[bool, str]:
        """
        Decide if elastic actor should switch from training to inference
        to help finish an ongoing rollout.
        
        Args:
            progress: Dict with 'work_queue_size', 'pending_tasks', 'active', etc.
            current_throughput: Current inference throughput (groups/sec)
        
        Returns:
            (should_switch: bool, reason: str)
        """
        # Check if rollout is even active
        if not progress.get('active', False):
            return False, "No active rollout"
        
        if progress.get('aborted', False):
            return False, "Rollout is aborted"
        
        # Calculate total remaining work
        queue_size = progress.get('work_queue_size', 0)
        pending_tasks = progress.get('pending_tasks', 0)
        total_remaining = queue_size + pending_tasks
        
        # Not enough work to justify switching
        if total_remaining < cls.MIN_WORK_THRESHOLD:
            return False, f"Not enough work remaining ({total_remaining} < {cls.MIN_WORK_THRESHOLD})"
        
        # Estimate time saved
        # With 1 engine: time = remaining / throughput
        # With 2 engines: time = remaining / (2 * throughput)
        # Time saved = time_with_1_engine - time_with_2_engines - switch_overhead
        
        time_with_one_engine = total_remaining / current_throughput
        time_with_two_engines = total_remaining / (2 * current_throughput)
        time_saved = time_with_one_engine - time_with_two_engines - cls.SWITCH_OVERHEAD_SECONDS
        
        if time_saved > 5:  # At least 5 seconds saved
            return True, (
                f"Worth switching: {total_remaining} groups remaining, "
                f"will save ~{time_saved:.1f}s "
                f"(queue={queue_size}, pending={pending_tasks})"
            )
        else:
            return False, (
                f"Not worth switching: would only save {time_saved:.1f}s "
                f"({total_remaining} groups, overhead={cls.SWITCH_OVERHEAD_SECONDS}s)"
            )

