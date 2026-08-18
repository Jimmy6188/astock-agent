"""
astock MCP HTTP 代理 - 将 stdio server 转为 HTTP 端点
让 Proma 能通过 HTTP MCP 类型连接 astock-agent

启动:
  python tools/mcp_http_proxy.py [--port 8001]

Proma mcp.json 配置:
  {
    "astock": {
      "type": "http",
      "enabled": true,
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
"""

import sys, os, asyncio, argparse

# 将项目 src 加入路径
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from mcp.server.streamable_http import StreamableHTTPServer
from mcp_server import server as astock_server

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8001)
parser.add_argument('--host', type=str, default='127.0.0.1')
args = parser.parse_args()

async def main():
    import uvicorn
    import starlette.applications

    from mcp.server.streamable_http_manager import StreamableHTTPManager

    manager = StreamableHTTPManager(astock_server)
    app = starlette.applications.Starlette()
    manager.install_routes(app)

    print(f'astock MCP HTTP 代理启动: http://{args.host}:{args.port}/mcp')
    print(f'Proma 配置: "url": "http://{args.host}:{args.port}/mcp"')
    await uvicorn.serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    asyncio.run(main())