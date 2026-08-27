/** Node parity check for S9Engine vs seven_numbers.json */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.join(__dirname, "..");
const engineSrc = fs.readFileSync(path.join(root, "widgets", "s9_engine.js"), "utf8");
const sandbox = { globalThis: {}, window: {}, console };
sandbox.window = sandbox.globalThis;
vm.runInNewContext(engineSrc, sandbox);
const S9 = sandbox.globalThis.S9Engine;

const bundle = JSON.parse(
  fs.readFileSync(path.join(root, "data", "evaluation", "widget_weights.json"), "utf8")
);
S9.setBundle(bundle);
const seven = JSON.parse(
  fs.readFileSync(path.join(root, "data", "evaluation", "seven_numbers.json"), "utf8")
);

const sentence = seven.user_sentence;
const r = S9.runPipeline(sentence, { chunkSize: 1024 });
const dl = Math.abs(r.loss0 - seven.loss0);
const dp = Math.abs(r.ppl0 - seven.ppl0);
console.log("loss0 js=", r.loss0.toFixed(6), "py=", seven.loss0.toFixed(6), "diff=", dl.toFixed(6));
console.log("ppl0 js=", r.ppl0.toFixed(4), "py=", seven.ppl0.toFixed(4), "diff=", dp.toFixed(4));
if (dl > 0.05 || dp > 2) {
  console.error("PARITY FAIL");
  process.exit(1);
}
console.log("PARITY OK");
process.exit(0);
