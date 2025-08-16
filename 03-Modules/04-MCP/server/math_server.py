
#!/usr/bin/env python
"""간단한 수학 연산을 제공하는 MCP 서버"""

from mcp.server.fastmcp import FastMCP
from typing import Annotated

# MCP 서버 인스턴스 생성
mcp = FastMCP("Math Operations")

@mcp.tool()
def add(
    a: Annotated[float, "첫 번째 숫자"],
    b: Annotated[float, "두 번째 숫자"]
) -> float:
    """두 숫자를 더합니다."""
    return a + b

@mcp.tool()
def multiply(
    a: Annotated[float, "첫 번째 숫자"],
    b: Annotated[float, "두 번째 숫자"]
) -> float:
    """두 숫자를 곱합니다."""
    return a * b

@mcp.tool()
def divide(
    a: Annotated[float, "분자"],
    b: Annotated[float, "분모"]
) -> float:
    """두 숫자를 나눕니다."""
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다")
    return a / b

@mcp.tool()
def power(
    base: Annotated[float, "밑수"],
    exponent: Annotated[float, "지수"]
) -> float:
    """거듭제곱을 계산합니다."""
    return base ** exponent

if __name__ == "__main__":
    # stdio 전송 방식으로 서버 실행
    mcp.run(transport="stdio")
