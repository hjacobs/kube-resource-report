#!/usr/bin/env python3
"""
This is an example application registry with test data.
Run this script and use --application-registry=http://localhost:8080
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

DATA = {
    "example-app": {"team_id": "example-team", "active": True},
    "legacy-service": {"team_id": "example-team", "active": False},
    "kube-ops-view": {"team_id": "hjacobs", "active": True},
    "kube-resource-report": {"team_id": "hjacobs", "active": True},
    "application-registry": {"team_id": "hjacobs", "active": True},
}


class HTTPServer_RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return

        elif not self.path.startswith("/apps/"):
            self.send_response(404)
            self.end_headers()
            return

        parts = self.path.split("/")
        application_id = parts[2]

        data = DATA.get(application_id)

        self.send_response(200 if data else 404)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        serialized = json.dumps(data)
        self.wfile.write(bytes(serialized, "utf-8"))


if __name__ == "__main__":
    server_address = ("0.0.0.0", 8080)
    httpd = HTTPServer(server_address, HTTPServer_RequestHandler)
    print("Listening on {}:{}..".format(*server_address))
    httpd.serve_forever()
