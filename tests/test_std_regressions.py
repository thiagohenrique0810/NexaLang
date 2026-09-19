"""Integration regressions for the standard library's native network paths."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StandardLibraryRegressions(unittest.TestCase):
    def setUp(self):
        self.compiler = shutil.which("clang")
        if not self.compiler:
            self.skipTest("clang unavailable")
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-std-tests-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def build(self, source, libraries=()):
        program = self.directory / "main.nxl"
        ir = self.directory / "main.ll"
        binary = self.directory / ("main.exe" if os.name == "nt" else "main")
        program.write_text(source)
        result = subprocess.run(
            [sys.executable, str(ROOT / "bootstrap/main.py"), str(program), "--out", str(ir), "--opt", "0"],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = [self.compiler, str(ir), str(ROOT / "runtime/nexa_async.c"), "-o", str(binary), *libraries]
        if os.name == "nt":
            command.append("-lws2_32")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return binary

    def test_vec_growth_and_explicit_drop(self):
        binary = self.build("""use std::vec::Vec;
fn main() -> i32 {
    let mut values = Vec::<i32>::new();
    for i in 0..40 { values.push(i * 3); }
    print(values.len());
    print(values.get(5).unwrap());
    print(values.get(39).unwrap());
    values.drop();
    return 0;
}
""")
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines(), ["40", "15", "117"])

    def test_hashmap_removal_preserves_collision_chain(self):
        binary = self.build("""use std::map::HashMap;
fn main() -> i32 {
    let mut map = HashMap::<i32,i32>::new();
    map.insert(1, 10);
    map.insert(17, 20);
    map.insert(33, 30);
    map.remove(1);
    print(map.get(17).unwrap());
    print(map.get(33).unwrap());
    map.remove(17);
    print(map.get(33).unwrap());
    print(map.len());
    return 0;
}
""")
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines(), ["20", "30", "30", "1"])

    def test_executor_accepts_eager_completed_tasks(self):
        binary = self.build("""use std::task::Executor;
use std::task::block_on;
extern "C" { fn malloc(size: i32) -> *u8; fn free(ptr: *u8); }
fn main() -> i32 {
    let handle = malloc(16);
    let done = cast::<*bool>(handle);
    *done = true;
    let result_ptr = cast::<*i32>(ptr_offset::<u8>(handle, 4));
    *result_ptr = 73;
    let mut executor = Executor::new();
    executor.spawn(handle);
    executor.run();
    print(block_on::<i32>(handle));
    free(handle);
    return 0;
}
""")
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "73")

    @unittest.skipIf(os.name == "nt", "libcurl installation is optional on Windows")
    def test_http_client_resets_method_and_header_ownership(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                requests.append((self.command, self.headers.get("Content-Type"), body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"posted")

            def do_GET(self):
                requests.append((self.command, self.headers.get("Content-Type"), b""))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"got")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"
        binary = self.build(f'''use std::net::HttpClient;
fn main() -> i32 {{
    let client = HttpClient::new();
    let first = client.post_json("{url}/first", "{{}}", 2);
    print(first.status);
    let second = client.get("{url}/second");
    print(second.status);
    print(second.body);
    return 0;
}}
''', ["-lcurl"])
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines(), ["200", "200", "got"])
        self.assertEqual(requests, [("POST", "application/json", b"{}"), ("GET", None, b"")])

    def test_http_server_accumulates_body_and_disables_cors(self):
        # Keep the selected port reserved during compilation, which can be slow
        # when the full suite is running. Release it immediately before spawn.
        reservation = socket.socket()
        self.addCleanup(reservation.close)
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        binary = self.build(f'''use std::http::HttpServer;
use std::http::Request;
use std::http::Response;
fn echo(req: &Request) -> Response {{
    if (req.content_length != 11) {{ return Response::text(500, "wrong body length"); }}
    return Response::text(200, req.body_str());
}}
fn main() -> i32 {{
    let mut server = HttpServer::new({port});
    server.set_cors(false);
    server.post("/echo", echo);
    server.run();
    return 0;
}}
''')
        reservation.close()
        process = subprocess.Popen([str(binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def stop_server():
            if process.poll() is None:
                process.terminate()
            try:
                return process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.communicate(timeout=5)

        # communicate also reaps the child and closes its pipes on failed asserts.
        self.addCleanup(stop_server)
        connection = None
        deadline = time.monotonic() + 10
        last_error = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.fail(f"server exited: {stdout!r} {stderr!r}")
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError as error:
                last_error = error
                time.sleep(0.02)
        if connection is None:
            stdout, stderr = stop_server()
            self.fail(f"server did not start within 10s on port {port}; "
                      f"last connection error: {last_error!r}; "
                      f"exit={process.returncode}; stdout={stdout!r}; stderr={stderr!r}")
        with connection:
            connection.sendall(b"POST /echo HTTP/1.1\r\nhost: localhost\r\ncontent-length: 11\r\n\r\nhello ")
            time.sleep(0.05)
            connection.sendall(b"world")
            response = bytearray()
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
        self.assertTrue(response.startswith(b"HTTP/1.1 200"), response)
        self.assertEqual(response.split(b"\r\n\r\n", 1)[1], b"hello world")
        self.assertNotIn(b"Access-Control-Allow-Origin", response)
        stop_server()


if __name__ == "__main__":
    unittest.main()
