import threading
import time

class InvariantViolation(RuntimeError):
    pass


class SystemInvariant:
    _lock = threading.Lock()
    _active_env = None
    _env_generation = 0

    @classmethod
    def register_env(cls, env):
        with cls._lock:
            if cls._active_env is not None:
                raise InvariantViolation(
                    "Invariant S1 violated: multiple LeanEnv in one worker"
                )
            cls._active_env = env
            cls._env_generation += 1
            env._generation = cls._env_generation

    @classmethod
    def unregister_env(cls, env):
        with cls._lock:
            if cls._active_env is env:
                cls._active_env = None

    @classmethod
    def check_env_alive(cls, env):
        if cls._active_env is not env:
            raise InvariantViolation(
                "Invariant S5 violated: using stale LeanEnv"
            )
