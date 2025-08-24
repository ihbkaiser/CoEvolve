import io
import sys
import contextlib

def evaluate(code: str, test_cases: list):
    logs = []
    all_passed = True
    for i, case in enumerate(test_cases, start=1):
        testtype = case.get("testtype", "stdin")  # Default to stdin for backward compatibility
        expected_output = str(case.get("output", "")).strip()
        passed = False
        output = ""
        try:
            if testtype == "stdin":
                input_data = case.get("input", "")
                # Ensure input() has a line to read (avoid EOFError on empty final line)
                if not input_data.endswith("\n"):
                    input_data += "\n"
                f_in = io.StringIO(input_data)
                f_out = io.StringIO()
                f_err = io.StringIO()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(f_out))
                    stack.enter_context(contextlib.redirect_stderr(f_err))
                    old_stdin = sys.stdin
                    sys.stdin = f_in
                    try:
                        # Run the user code in a clean, script-like globals
                        exec_globals = {"__name__": "__main__"}
                        exec(code, exec_globals)
                    finally:
                        # Always restore stdin even if exec() raises
                        sys.stdin = old_stdin
                out = f_out.getvalue().strip()
                err = f_err.getvalue().strip()
                output = out
                passed = (output == expected_output)
            elif testtype == "functional":
                exec_globals = {}
                exec(code, exec_globals)
                if "main" not in exec_globals:
                    raise RuntimeError("No function named 'main' found in code.")
                main_fn = exec_globals["main"]
                args = eval(case["input"])
                if not isinstance(args, tuple):
                    args = (args,)
                output = str(main_fn(*args)).strip()
                passed = (output == expected_output)
            else:
                raise ValueError(f"Unknown testtype: {testtype}")
        except Exception as e:
            if testtype == "stdin":
                out = f_out.getvalue().strip() if 'f_out' in locals() else ""
                err = f_err.getvalue().strip() if 'f_err' in locals() else ""
                output = (out + ("\n" if out and err else "") + err) or f"Error: {e}"
            else:
                output = f"Error: {e}"
        if passed:
            logs.append(f"test {i} passed")
        else:
            logs.append(f"test {i} failed (expected: {expected_output}, got: {output})")
            all_passed = False
    return "\n".join(logs), all_passed