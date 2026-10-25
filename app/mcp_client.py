from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MCP_DATA_DIR = (
    r"C:\Users\GR131235\Desktop"
    r"\Enterprise-Knowledge-Assistant\mcp_data"
)


server_params = StdioServerParameters(
    command="npx.cmd",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        MCP_DATA_DIR,
    ],
)


async def read_mcp_file(file_path: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "read_text_file",
                {
                    "path": file_path,
                },
            )

            return result