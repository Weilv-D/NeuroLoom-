import { expect, test } from "@playwright/test";

test.describe("NeuroLoom Qwen starfield", () => {
  test("loads the single-model workspace", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: /Qwen3\.5-0\.8B, rendered as a live starfield\./ })).toBeVisible();
    await page.waitForSelector(".scene-stage canvas");
    await expect(page.locator(".workspace")).toHaveScreenshot("qwen-starfield-workspace.png");
  });
});
