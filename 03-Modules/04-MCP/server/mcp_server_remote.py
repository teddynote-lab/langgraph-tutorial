from fastmcp import FastMCP
from typing import Optional
import pytz
from datetime import datetime

mcp = FastMCP(
    "Current Time",  # Name of the MCP server
    instructions="Information about the current time in a given timezone",  # Instructions for the LLM on how to use this tool
)


@mcp.tool()
async def get_current_time(timezone: Optional[str] = "Asia/Seoul") -> str:
    """
    Get current time information for the specified timezone.

    This function returns the current system time for the requested timezone.

    Args:
        timezone (str, optional): The timezone to get current time for. Defaults to "Asia/Seoul".

    Returns:
        str: A string containing the current time information for the specified timezone
    """
    try:
        # Get the timezone object
        tz = pytz.timezone(timezone)

        # Get current time in the specified timezone
        current_time = datetime.now(tz)

        # Format the time as a string
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        return f"Current time in {timezone} is: {formatted_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Please provide a valid timezone."
    except Exception as e:
        return f"Error getting time: {str(e)}"


if __name__ == "__main__":
    # Print a message indicating the server is starting
    print("mcp remote server is running...")

    # start the server
    mcp.run(transport="streamable-http", port=8002)
