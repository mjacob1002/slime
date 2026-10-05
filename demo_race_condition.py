"""
Demonstration of the race condition and how await fixes it
"""
import asyncio

# Global state
pendings = set()
work_queue = asyncio.Queue()


async def worker(worker_id: int):
    """Simulates a worker that pulls from queue"""
    print(f"  Worker {worker_id}: Waiting for work...")
    try:
        item = await asyncio.wait_for(work_queue.get(), timeout=0.1)
        print(f"  Worker {worker_id}: Got item '{item}', creating task")
        
        # Simulate adding task to pendings
        task = asyncio.create_task(asyncio.sleep(1))
        pendings.add(task)
        
        print(f"  Worker {worker_id}: Added task to pendings. Total: {len(pendings)}")
    except asyncio.TimeoutError:
        print(f"  Worker {worker_id}: Timed out")


def submit_tasks_sync(items):
    """WITHOUT await - causes race condition"""
    print("submit_tasks_sync: Adding items to queue...")
    for item in items:
        work_queue.put_nowait(item)
    print("submit_tasks_sync: Done adding items (synchronous)")
    # No yield here - returns immediately


async def submit_tasks_async(items):
    """WITH await - prevents race condition"""
    print("submit_tasks_async: Adding items to queue...")
    for item in items:
        work_queue.put_nowait(item)
    print("submit_tasks_async: Yielding to event loop...")
    await asyncio.sleep(0)  # ← The magic line
    print("submit_tasks_async: Returned from yield")


async def main_with_race():
    """Demonstrates the RACE CONDITION"""
    print("\n" + "="*60)
    print("SCENARIO 1: WITHOUT await (Race Condition)")
    print("="*60 + "\n")
    
    # Clear state
    pendings.clear()
    while not work_queue.empty():
        work_queue.get_nowait()
    
    # Start workers
    workers = [asyncio.create_task(worker(i)) for i in range(3)]
    
    # Give workers a moment to start waiting
    await asyncio.sleep(0.01)
    
    print("Main: Submitting tasks (synchronous)...")
    submit_tasks_sync(["item1", "item2", "item3"])
    
    print(f"Main: Checking pendings immediately... len={len(pendings)}")
    if not pendings:
        print("Main: ❌ RACE CONDITION! No tasks in pendings yet!")
    else:
        print("Main: ✓ Tasks found in pendings")
    
    # Wait for workers to actually run
    await asyncio.sleep(0.1)
    print(f"Main: After waiting 100ms... len={len(pendings)}")
    
    # Cleanup
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


async def main_without_race():
    """Demonstrates NO RACE with await"""
    print("\n" + "="*60)
    print("SCENARIO 2: WITH await (No Race)")
    print("="*60 + "\n")
    
    # Clear state
    pendings.clear()
    while not work_queue.empty():
        work_queue.get_nowait()
    
    # Start workers
    workers = [asyncio.create_task(worker(i)) for i in range(3)]
    
    # Give workers a moment to start waiting
    await asyncio.sleep(0.01)
    
    print("Main: Submitting tasks (async with yield)...")
    await submit_tasks_async(["item1", "item2", "item3"])
    
    print(f"Main: Checking pendings immediately... len={len(pendings)}")
    if not pendings:
        print("Main: ❌ No tasks in pendings!")
    else:
        print("Main: ✓ Tasks found in pendings (no race!)")
    
    # Cleanup
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


async def main():
    await main_with_race()
    await main_without_race()
    print("\n" + "="*60)
    print("Summary:")
    print("  - Without await: Race condition occurs")
    print("  - With await: Workers run before check")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

