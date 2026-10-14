import { chromium } from "/opt/node-tools/node_modules/playwright/index.mjs";
import fs from "fs"; import path from "path";
const [,, htmlPath, outDir, mode] = process.argv;
const body = fs.readFileSync(htmlPath, "utf8");
const page_html = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${body}</body></html>`;
const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium", args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: mode === "page" ? { width: 420, height: 1400 } : { width: 1200, height: 750 }, deviceScaleFactor: mode === "page" ? 2 : 1 });
page.on("console", m => console.log("console:", m.text()));
page.on("pageerror", e => console.log("pageerror:", e.message));
await page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**", r => {
  const rel = r.request().url().split("three@0.160.0/")[1];
  r.fulfill({ path: path.resolve("node_modules/three", rel), contentType: "application/javascript" });
});
await page.route("https://fonts.googleapis.com/**", r => r.abort());
await page.route("http://local/", r => r.fulfill({ body: page_html, contentType: "text/html" }));
await page.goto("http://local/");
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
if (mode === "page") {
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outDir, "page.png"), fullPage: true });
} else {
  await page.evaluate(() => document.body.classList.add("capture"));
  const shots = JSON.parse(fs.readFileSync(path.join(outDir, "shots.json"), "utf8"));
  for (const s of shots) {
    await page.evaluate(o => window.__shot(o), s);
    await page.waitForTimeout(300);
    await page.locator("#stage").screenshot({ path: path.join(outDir, s.file) });
    console.log("shot", s.file);
  }
}
await browser.close();
