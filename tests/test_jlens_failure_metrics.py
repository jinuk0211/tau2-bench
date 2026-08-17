from tau2.agent.jlens_failure_metrics import simulation_failure_metrics


def test_simulation_metrics_combine_reward_review_and_structural_failures():
    simulation = {
        "reward": 0,
        "termination_reason": "too_many_errors",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "a", "name": "lookup", "arguments": {"id": 1}}],
            },
            {
                "role": "tool",
                "id": "a",
                "requestor": "assistant",
                "content": "not found",
                "error": True,
            },
            {
                "role": "assistant",
                "tool_calls": [{"id": "b", "name": "lookup", "arguments": {"id": 1}}],
            },
        ],
        "review": {
            "errors": [
                {
                    "source": "agent",
                    "error_tags": ["wrong_sequence", "tool_error"],
                },
                {"source": "user", "error_tags": ["incorrect_information"]},
            ]
        },
    }

    metrics = simulation_failure_metrics(simulation)
    assert not metrics["task_success"]
    assert metrics["agent_review_error_count"] == 1
    assert metrics["user_review_error_count"] == 1
    assert metrics["review_error_counts_by_tag"] == {
        "incorrect_information": 1,
        "tool_error": 1,
        "wrong_sequence": 1,
    }
    assert metrics["tool_call_error_count"] == 1
    assert metrics["immediate_repeat_tool_call_count"] == 1
    assert metrics["error_termination"]
