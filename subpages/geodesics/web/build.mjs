// Assemble the standalone geodesics.html from shell.html + the four modules.
// Run: node build.mjs
import { readFileSync, writeFileSync } from "fs";

const read = (f) => readFileSync(f, "utf8");
let html = read("shell.html");
const parts = { "/*__CORE__*/": "core.js", "/*__CELLS__*/": "cells.js",
                "/*__AI__*/": "ai.js", "/*__APP__*/": "app.js" };
for (const [marker, file] of Object.entries(parts)) {
  if (!html.includes(marker)) throw new Error("missing marker " + marker);
  html = html.split(marker).join(read(file));
}
writeFileSync("geodesics.html", html);
console.log("geodesics.html assembled:", html.length, "bytes");
