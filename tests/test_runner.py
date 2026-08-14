from pathlib import Path

from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.runner import solve_task

FIXTURE = Path("tasks/workshop/01-shopcart")
FIXED_CART = """from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices))
"""


def test_solve_task_wires_everything_together():
    llm = FakeLLMClient(
        [
            assistant_tool_call("run_tests", {}, call_id="c1"),
            assistant_tool_call(
                "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c2"
            ),
            assistant_tool_call("run_tests", {}, call_id="c3"),
            assistant_text("done"),
        ]
    )

    result = solve_task(FIXTURE, llm=llm)

    assert result.solved is True
    assert result.task_id == "01-shopcart"
