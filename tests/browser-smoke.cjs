const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.env.TEST_BASE_URL || "http://127.0.0.1:8812/";
const imagePath = process.env.TEST_IMAGE;
const executablePath = process.env.CHROME_PATH;
const screenshotDir = process.env.SCREENSHOT_DIR;

if (!imagePath) throw new Error("TEST_IMAGE is required");
if (!executablePath) throw new Error("CHROME_PATH is required");

(async () => {
  if (screenshotDir) fs.mkdirSync(screenshotDir, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath });
  const page = await browser.newPage({ viewport: { width: 430, height: 932 } });
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelector("#modelStatus")?.textContent?.includes("已准备好"));
    if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, "01-mobile-home.png"), fullPage: true });
    await page.setInputFiles("#photoInput", imagePath);
    assert.equal(await page.locator("#introCard").isVisible(), false);
    assert.equal(await page.locator("#uploadCard").isVisible(), false);
    assert.equal(await page.locator("#previewCard").isVisible(), true);
    assert.equal(await page.locator("#analyzeBtn").isEnabled(), true);
    if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, "02-mobile-preview.png"), fullPage: true });

    await page.click("#analyzeBtn");
    await page.waitForFunction(
      () => !document.querySelector("#report")?.classList.contains("hidden"),
      null,
      { timeout: 90_000 },
    );

    assert.equal(await page.locator("#introCard").isVisible(), false);
    assert.equal(await page.locator("#uploadCard").isVisible(), false);
    assert.equal(await page.locator("#previewCard").isVisible(), false);
    assert.equal(await page.locator("#resultHero").isVisible(), true);

    const result = {
      score: await page.locator("#totalScore").textContent(),
      style: await page.locator("#comboName").textContent(),
      identity: await page.locator("#identityLine").textContent(),
      dimensions: await page.locator("#dimensionOverview .dimension-row").count(),
      highlights: await page.locator("#insightGrid .insight-card").count(),
      sections: await page.locator("#analysisGrid .analysis-card").count(),
      advice: await page.locator("#adviceGrid .advice-card").count(),
      consoleErrors,
    };
    assert.equal(result.dimensions, 4);
    assert.equal(result.highlights, 3);
    assert.equal(result.sections, 4);
    assert.equal(result.advice, 3);
    assert.match(result.style, / × /);
    assert.ok(result.identity.length > 4);
    assert.deepEqual(result.consoleErrors, []);
    if (screenshotDir) {
      await page.screenshot({ path: path.join(screenshotDir, "03-mobile-report.png"), fullPage: true });
      await page.setViewportSize({ width: 1365, height: 1000 });
      await page.screenshot({ path: path.join(screenshotDir, "04-desktop-report.png"), fullPage: true });
      await page.setViewportSize({ width: 430, height: 932 });
    }

    await page.click("#shareBtn");
    assert.equal(await page.locator("#shareModal").isVisible(), true);
    const canvasSizes = await page.locator(".share-card-canvas").evaluateAll((canvases) =>
      canvases.map((canvas) => ({ width: canvas.width, height: canvas.height })),
    );
    assert.equal(canvasSizes.length, 5);
    assert.ok(canvasSizes.every(({ width, height }) => width === 1080 && height === 1440));
    assert.equal(await page.locator(".share-card-canvas.active").count(), 1);
    const visibleCardBeforePrivacy = await page.locator(".share-card-canvas.active").evaluate((canvas) => canvas.toDataURL("image/png"));
    if (screenshotDir) {
      const cardDataUrls = await page.locator(".share-card-canvas").evaluateAll((canvases) =>
        canvases.map((canvas) => canvas.toDataURL("image/png")),
      );
      cardDataUrls.forEach((dataUrl, index) => {
        fs.writeFileSync(
          path.join(screenshotDir, `05-share-card-${index + 1}.png`),
          Buffer.from(dataUrl.split(",")[1], "base64"),
        );
      });
      await page.screenshot({ path: path.join(screenshotDir, "06-mobile-share-studio.png"), fullPage: true });
    }
    await page.check("#hidePhotoToggle");
    const visibleCardAfterPrivacy = await page.locator(".share-card-canvas.active").evaluate((canvas) => canvas.toDataURL("image/png"));
    assert.notEqual(visibleCardBeforePrivacy, visibleCardAfterPrivacy);
    await page.click("#nextCardBtn");
    assert.equal(await page.locator("#cardCounter").textContent(), "2 / 5");
    await page.click("#closeShareBtn");
    assert.equal(await page.locator("#shareModal").isVisible(), false);
    assert.deepEqual(consoleErrors, []);

    await page.click("#retestTopBtn");
    assert.equal(await page.locator("#report").isVisible(), false);
    assert.equal(await page.locator("#uploadCard").isVisible(), true);
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
