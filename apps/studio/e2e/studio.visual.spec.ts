import { expect, type Page, test } from "@playwright/test";

test.describe("NeuroLoom visual baselines", () => {
  test("MLP story workspace", async ({ page }) => {
    await openTrace(page, "Spiral MLP", "Spiral MLP");
    await expect(page.locator(".workspace")).toHaveScreenshot("mlp-story-workspace.png");
  });

  test("CNN studio workspace", async ({ page }) => {
    await openTrace(page, "Fashion CNN", "Fashion-MNIST CNN");
    await page.getByRole("button", { name: "Studio Mode" }).click();
    await page.waitForTimeout(900);
    await expect(page.locator(".workspace")).toHaveScreenshot("cnn-studio-workspace.png");
  });

  test("Transformer studio workspace", async ({ page }) => {
    await openTrace(page, "Tiny GPT Transformer", "Tiny GPT-style Transformer");
    await page.getByRole("button", { name: "Studio Mode" }).click();
    await page
      .getByTestId(/attention-token-/)
      .filter({ hasText: "glows" })
      .click();
    await page.waitForTimeout(900);
    await expect(page.locator(".workspace")).toHaveScreenshot("transformer-studio-workspace.png");
  });
});

async function openTrace(page: Page, cardLabel: string, titleLabel: string) {
  await page.goto("/");
  await page.waitForSelector(".stage-frame canvas");

  if (cardLabel !== "Spiral MLP") {
    await page.getByRole("button", { name: cardLabel }).click();
  }

  await expect(page.locator(".toolbar__group--meta .meta-pill").first()).toContainText(titleLabel);
  await page.waitForTimeout(1200);
}
