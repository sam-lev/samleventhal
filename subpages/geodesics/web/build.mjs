// Assemble the standalone geodesics.html from shell.html + the four modules,
// plus every model in models/*.js (each wrapped with a WEIGHTS constant taken
// from a sibling <name>.weights.json, or null). Run: node build.mjs
import { readFileSync, writeFileSync, readdirSync, existsSync } from "fs";

const read = (f) => readFileSync(f, "utf8");
let html = read("shell.html");
const parts = { "/*__CORE__*/": "core.js", "/*__CELLS__*/": "cells.js",
                "/*__AI__*/": "ai.js", "/*__APP__*/": "app.js" };
for (const [marker, file] of Object.entries(parts)) {
  if (!html.includes(marker)) throw new Error("missing marker " + marker);
  html = html.split(marker).join(read(file));
}

let modelsJs = "// no models/ directory";
if (existsSync("models")) {
  const files = readdirSync("models").filter(f => f.endsWith(".js")).sort();
  modelsJs = files.map(f => {
    const wf = "models/" + f.replace(/\.js$/, ".weights.json");
    const w = existsSync(wf) ? read(wf).trim() : "null";
    return "// ---- models/" + f + " ----\n(function () {\n" +
           "const WEIGHTS = " + w + ";\n" + read("models/" + f) + "\n})();";
  }).join("\n");
  console.log("bundled models:", files.length ? files.join(", ") : "(none)");
}
if (!html.includes("/*__MODELS__*/")) throw new Error("missing marker /*__MODELS__*/");
html = html.split("/*__MODELS__*/").join(modelsJs);

const out = process.argv[2] || "geodesics.html";
writeFileSync(out, html);
console.log(out, "assembled:", html.length, "bytes");
