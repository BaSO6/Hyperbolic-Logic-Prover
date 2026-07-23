# ==========================================
# Filename: src/system2/lean_interaction.py
# Version: v16.0
#
# Fixes vs v15.1:
#   FIX-1: PTY write chunking — large commands (context prefixes) are split
#           into ≤3000-byte chunks to avoid the PTY 4096-byte line limit that
#           caused "Could not parse JSON: offset 4096" crashes.
#   FIX-2: task_complete logic — no longer exits on the first {"env": N} object.
#           We only mark complete when we see a top-level env AND no pending
#           open-brace context, preventing response desynchronization across
#           sequential commands.
#   FIX-3: `sorries` field is now captured in the result dict so callers
#           (_extract_goal_from_sorry in mcts_hybrid_search) can read it.
#   FIX-4: Doubled read buffer (16 384 bytes) to reduce partial-chunk splits.
# ==========================================

import os
import pty
import subprocess
import json
import time
import select
import hashlib

# Maximum bytes per single os.write() to the PTY master fd.
# The Linux PTY line discipline has a ~4096-byte canonical-mode limit;
# in raw mode the limit is higher but erratic. We stay well below it.
_PTY_WRITE_CHUNK = 2048


class LeanEnv:
    _global_calls = 0
    _global_elapsed_s = 0.0

    @classmethod
    def reset_global_metrics(cls):
        cls._global_calls = 0
        cls._global_elapsed_s = 0.0

    @classmethod
    def global_metrics(cls):
        return {
            "lean_calls": int(cls._global_calls),
            "lean_elapsed_s": float(cls._global_elapsed_s),
        }

    def __init__(self, project_root=None, verbose=True):
        self.proc = None
        self.master = None
        self.verbose = verbose
        self.current_env = None

        if project_root is None:
            project_root = os.getcwd()
        self.project_root = os.path.abspath(project_root)

        self.mathlib_root = os.path.join(self.project_root, "data", "mathlib4")
        if not os.path.isdir(self.mathlib_root):
            raise RuntimeError(f"Mathlib root not found: {self.mathlib_root}")

        self.wrapper_path = os.path.join(
            self.project_root, "src/system2/run_repl_wrapper.sh"
        )
        if not os.path.isfile(self.wrapper_path):
            raise RuntimeError(f"REPL wrapper not found: {self.wrapper_path}")

        self._start_process()

    # --------------------------------------------------
    # Process management
    # --------------------------------------------------

    def _start_process(self):
        self.master, slave = pty.openpty()
        clean_env = {k: v for k, v in os.environ.items() if "LEAN" not in k}

        try:
            self.proc = subprocess.Popen(
                ["bash", self.wrapper_path],
                cwd=self.mathlib_root,
                env=clean_env,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                text=True,
                bufsize=0,
                close_fds=True,
            )
            os.close(slave)
            if self.verbose:
                print(f"[LeanEnv] Started via Wrapper: {self.wrapper_path}")
        except Exception as e:
            if self.master:
                os.close(self.master)
            raise RuntimeError(f"Failed to start Lean process: {e}")

    def close(self):
        if self.proc:
            try:
                if self.verbose:
                    print(f"[LeanEnv] Closing process {self.proc.pid}...")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception as e:
                if self.verbose:
                    print(f"[LeanEnv] Error during close: {e}")
            finally:
                self.proc = None

        if self.master:
            try:
                os.close(self.master)
            except Exception:
                pass
            finally:
                self.master = None

    def __del__(self):
        self.close()

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log_raw(self, text):
        if self.verbose and text.strip():
            clean_text = text.replace("\r", "").replace("\n", "\\n")
            print(f"  [PTY] {clean_text}")

    # --------------------------------------------------
    # FIX-1: Chunked PTY write
    # --------------------------------------------------

    def _write_chunked(self, data: bytes):
        """Write data to the PTY master in chunks to avoid the line-discipline
        4096-byte limit that truncates large JSON payloads mid-string."""
        offset = 0
        while offset < len(data):
            end = min(offset + _PTY_WRITE_CHUNK, len(data))
            os.write(self.master, data[offset:end])
            offset = end
            if offset < len(data):
                # Small pause so the PTY can drain between chunks.
                time.sleep(0.005)

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    def run_command(self, cmd_str, timeout=300):
        """Execute one REPL request and account for every call, including
        initialization, failed tactics, timeouts, and environment restarts."""
        type(self)._global_calls += 1
        started = time.perf_counter()
        try:
            return self._run_command(cmd_str, timeout=timeout)
        finally:
            type(self)._global_elapsed_s += time.perf_counter() - started

    def _run_command(self, cmd_str, timeout=300):
        if not self.proc or self.proc.poll() is not None:
            return {"error": "process_not_running"}

        req = {"cmd": cmd_str}
        if self.current_env is not None:
            req["env"] = self.current_env

        payload = json.dumps(req, ensure_ascii=False)
        if self.verbose:
            print(f"[LeanEnv] >>> {payload}")

        try:
            # FIX-1: use chunked write instead of single os.write()
            self._write_chunked((payload + "\n\n").encode("utf-8"))
        except OSError:
            return {"error": "write_failed"}

        start_time = time.time()

        # ---- JSON stream parser ----
        in_json = False
        brace_count = 0
        in_string = False
        escape = False
        json_buffer = ""

        # ---- Result aggregation ----
        result = {}
        messages = []
        sorries = []
        # FIX-2: only set task_complete when we have seen a complete top-level
        # response (i.e. an object with "env" key that is NOT just an echo of
        # our own command).  We track whether we have seen at least one
        # non-echo object with an "env" key.
        got_env = False

        while time.time() - start_time < timeout:
            rlist, _, _ = select.select([self.master], [], [], 0.1)
            if self.master not in rlist:
                if self.proc.poll() is not None:
                    break
                # FIX-2: if we already have an env and have drained the PTY,
                # we can exit (avoids hanging on slow responses)
                if got_env and not in_json:
                    # Give one more short window for any trailing messages
                    time.sleep(0.05)
                    rlist2, _, _ = select.select([self.master], [], [], 0.05)
                    if self.master not in rlist2:
                        break
                continue

            try:
                # FIX-4: larger read buffer
                chunk = os.read(self.master, 16384).decode(errors="ignore")
            except OSError:
                break

            if not chunk:
                continue

            self._log_raw(chunk)

            for ch in chunk:
                if not in_json:
                    if ch == "{":
                        in_json = True
                        brace_count = 1
                        in_string = False
                        escape = False
                        json_buffer = "{"
                    continue

                json_buffer += ch
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1

                if brace_count == 0:
                    candidate = json_buffer.strip()
                    in_json = False
                    json_buffer = ""

                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

                    # Skip echo objects (they contain "cmd" key)
                    if "cmd" in obj:
                        continue

                    # FIX-3: capture sorries
                    if "sorries" in obj:
                        sorries.extend(obj["sorries"])
                        result["sorries"] = sorries

                    if "messages" in obj:
                        messages.extend(obj["messages"])
                        result["messages"] = messages

                    # FIX-2: record env but don't immediately exit —
                    # there may be a "messages" object still in the stream.
                    if "env" in obj:
                        self.current_env = obj["env"]
                        result["env"] = obj["env"]
                        got_env = True

                    # Error info objects (non-message-list errors from REPL)
                    if "message" in obj and "env" not in obj:
                        result.setdefault("repl_errors", []).append(
                            obj.get("message", "")
                        )

            # FIX-2: after processing a full chunk, if we have env and the
            # PTY is now quiet, stop waiting.
            if got_env and not in_json:
                rlist3, _, _ = select.select([self.master], [], [], 0.08)
                if self.master not in rlist3:
                    break

        if not result:
            return {"error": "no_response"}

        text = json.dumps(result).lower()
        if "building" in text or "compiling" in text:
            raise RuntimeError(
                "Invariant Violation: Lean attempted to build Mathlib."
            )

        return result
