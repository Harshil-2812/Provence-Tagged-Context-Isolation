"""
Ollama worker subprocess for executing LLM queries with hard process-level wall-clock deadlines.
Can be executed as: python -m agent.ollama_worker
Reads JSON from stdin, performs HTTP POST to Ollama, and writes JSON to stdout.
"""
import sys
import json
import urllib.request
import urllib.error


def run_worker():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            sys.stderr.write("No input provided to worker\n")
            sys.exit(1)
        req_data = json.loads(raw_input)
        url = req_data["url"]
        payload = req_data["payload"]
        socket_timeout = float(req_data.get("socket_timeout", 180.0))

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=socket_timeout) as resp:
            raw_resp = resp.read()

        response_obj = json.loads(raw_resp.decode("utf-8"))
        sys.stdout.write(json.dumps({"status": "ok", "data": response_obj}))
        sys.stdout.flush()
        sys.exit(0)
    except urllib.error.URLError as e:
        is_timeout = (
            isinstance(getattr(e, "reason", None), TimeoutError)
            or "timed out" in str(e).lower()
        )
        err_type = "TimeoutError" if is_timeout else "RuntimeError"
        sys.stdout.write(json.dumps({
            "status": "error",
            "error_type": err_type,
            "error_message": str(e)
        }))
        sys.stdout.flush()
        sys.exit(0)
    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({
            "status": "error",
            "error_type": "ValueError",
            "error_message": f"Malformed JSON from Ollama: {e}"
        }))
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        sys.stdout.write(json.dumps({
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }))
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    run_worker()
