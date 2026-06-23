#!/usr/bin/env node
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const host = process.env.HOST || "127.0.0.1";
const port = Number.parseInt(process.env.PORT || "8088", 10);

const types = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".webmanifest": "application/manifest+json; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
};

function resolvePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const cleanPath = normalize(decoded === "/" ? "/index.html" : decoded);
  const filePath = resolve(join(root, cleanPath));
  if (filePath !== root && !filePath.startsWith(`${root}${sep}`)) return null;
  if (existsSync(filePath) && statSync(filePath).isFile()) return filePath;
  return resolve(join(root, "404.html"));
}

const server = createServer((req, res) => {
  if (!req.url) {
    res.writeHead(400).end("Bad request");
    return;
  }

  const filePath = resolvePath(req.url);
  if (!filePath) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  const status = filePath.endsWith(`${sep}404.html`) && !req.url.includes("404.html") ? 404 : 200;
  res.writeHead(status, {
    "Content-Type": types[extname(filePath)] || "application/octet-stream",
    "Cache-Control": "no-store",
  });
  createReadStream(filePath).pipe(res);
});

server.listen(port, host, () => {
  console.log(`8 Bit Labs site preview: http://${host}:${port}`);
});
