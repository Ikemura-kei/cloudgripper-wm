import queue


class RobotPool:
    """Thread-safe singleton pool that assigns robot names to env instances.

    Call ``RobotPool.configure(["robot23"])`` once before creating any envs.
    Each ``CloudGripperEnv.__init__`` calls ``acquire()``; ``close()`` calls
    ``release()``. Works identically for 1 or N robots.
    """

    _queue: queue.Queue = queue.Queue()
    _configured: bool = False

    @classmethod
    def configure(cls, robot_names: list[str]) -> None:
        """Populate the pool. Replaces any previous configuration."""
        while not cls._queue.empty():
            try:
                cls._queue.get_nowait()
            except queue.Empty:
                break
        for name in robot_names:
            cls._queue.put(name)
        cls._configured = True

    @classmethod
    def acquire(cls) -> str:
        """Claim a robot name. Blocks up to 30 s, then raises if none available."""
        if not cls._configured:
            raise RuntimeError("RobotPool not configured — call RobotPool.configure() first.")
        try:
            return cls._queue.get(block=True, timeout=30)
        except queue.Empty:
            raise RuntimeError("No robots available in pool — all are in use or pool is too small.")

    @classmethod
    def release(cls, name: str) -> None:
        """Return a robot name to the pool."""
        cls._queue.put(name)

    @classmethod
    def reset(cls) -> None:
        """Drain the pool and clear configured state. Used in tests."""
        while not cls._queue.empty():
            try:
                cls._queue.get_nowait()
            except queue.Empty:
                break
        cls._configured = False
