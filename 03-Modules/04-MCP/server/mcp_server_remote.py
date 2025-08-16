from fastmcp import FastMCP

mcp = FastMCP(
    "Weather",  # Name of the MCP server
    instructions="You are a weather assistant that can answer questions about the weather in a given location.",  # Instructions for the LLM on how to use this tool
    host="0.0.0.0",  # Host address (0.0.0.0 allows connections from any IP)
    port=8005,  # Port number for the server
)


@mcp.tool()
async def get_weather(location: str) -> str:
    """
    Get current weather information for the specified location.

    This function simulates a weather service by returning a fixed response.
    In a production environment, this would connect to a real weather API.

    Args:
        location (str): The name of the location (city, region, etc.) to get weather for

    Returns:
        str: A string containing the weather information for the specified location
    """
    # Return a mock weather response
    # In a real implementation, this would call a weather API
    return f"It's always Sunny in {location}"


if __name__ == "__main__":
    # Print a message indicating the server is starting
    print("mcp remote server is running...")

    # Start the MCP server with SSE transport
    # Server-Sent Events (SSE) transport allows the server to communicate with clients
    # over HTTP, making it suitable for remote/distributed deployments
    mcp.run(transport="streamable-http")
