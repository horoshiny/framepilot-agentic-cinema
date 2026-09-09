"""Production entry point for FramePilot.

Replit supplies PORT for published deployments. The local default keeps the
development preview on the workspace's standard web port.
"""

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )