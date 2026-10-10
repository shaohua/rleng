from trajectory import final_answer, grounded, same_args, tool_calls, tool_results


def check(traj: dict) -> tuple[bool, str]:
    exp = traj["expected"]
    calls, results = tool_calls(traj), tool_results(traj)
    if not calls:
        return False, "answered without calling a tool"
    if calls[0]["tool"] != exp["tool"]:
        return False, f"called {calls[0]['tool']}, expected {exp['tool']}"
    if not same_args(calls[0]["args"], exp["args"]):
        return False, f"wrong args {calls[0]['args']}"
    if not results or not grounded(final_answer(traj), results[0]):
        return False, "final answer not grounded in tool result"
    return True, "ok"
