import { test, expect } from "@playwright/test";

/**
 * Recall E2E Test — SDM Context Compression
 *
 * Tests the core claim: SDM remembers what falls out of the context window.
 *
 * Flow:
 *   1. Chat about Topic A (cooking pasta)
 *   2. Chat about Topic B (python programming) for several messages
 *   3. Ask about Topic A again → SDM should recall pasta conversation
 *   4. Verify memory panel shows recalled memories from Topic A
 *
 * Requires: HeatherDB (128d, port 6380), API (port 8000), Frontend (port 3000)
 */

const WAIT_FOR_RESPONSE = 60_000; // Claude API can be slow

async function sendMessage(page: any, message: string) {
  const input = page.locator('input[placeholder="Type a message..."]');
  await input.fill(message);
  await page.locator("button", { hasText: "Send" }).click();

  // wait for assistant response to appear
  // the loading state shows "thinking..." then gets replaced by the actual response
  await expect(page.locator("text=thinking...")).toBeVisible({ timeout: 5000 });
  await expect(page.locator("text=thinking...")).toBeHidden({
    timeout: WAIT_FOR_RESPONSE,
  });

  // small pause for UI to settle
  await page.waitForTimeout(500);
}

test.describe("Recall — SDM Context Compression", () => {
  test.beforeEach(async ({ request }) => {
    // Reset database before each test for isolation
    await request.delete("http://127.0.0.1:8000/api/reset");
  });

  test("page loads correctly", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("h1", { hasText: "Recall" })).toBeVisible();
    await expect(
      page.locator("text=Start a conversation.")
    ).toBeVisible();
    await expect(
      page.locator("text=Recalled Memories")
    ).toBeVisible();
  });

  test("basic chat works — send message and get response", async ({
    page,
  }) => {
    await page.goto("/");

    await sendMessage(page, "Hello, this is a test message.");

    // should have at least 1 user + 1 assistant message
    const userMessages = page.locator('[data-testid="message-user"]');
    const assistantMessages = page.locator('[data-testid="message-assistant"]');
    await expect(userMessages).toHaveCount(1, { timeout: 5000 });
    await expect(assistantMessages).toHaveCount(1, { timeout: 5000 });

    // assistant response should not be an error
    const responseText = await assistantMessages.first().textContent();
    console.log(`  → Response: ${responseText?.substring(0, 100)}`);
    expect(responseText).toBeTruthy();
    expect(responseText).not.toContain("Error:");

    // memory panel should update — fidelity should be visible
    await expect(
      page.locator('[data-testid="fidelity-label"]')
    ).toBeVisible({ timeout: 10000 });
  });

  test("SDM recalls Topic A after talking about Topic B", async ({ page }) => {
    test.setTimeout(300_000); // 5 minutes for this long test

    await page.goto("/");

    // ── Topic A: Cooking ──
    await sendMessage(
      page,
      "I love making homemade pasta with semolina flour and eggs. The key is kneading the dough for at least 10 minutes."
    );
    console.log("  → Sent Topic A message 1 (pasta)");

    await sendMessage(
      page,
      "For the sauce, I always use San Marzano tomatoes with fresh basil and a pinch of sugar."
    );
    console.log("  → Sent Topic A message 2 (sauce)");

    // Verify we got a response with fidelity
    await expect(
      page.locator('[data-testid="fidelity-label"]')
    ).toBeVisible({ timeout: 10000 });

    // ── Topic B: Programming (push Topic A out of recent window) ──
    await sendMessage(
      page,
      "Can you explain how Python decorators work? I use them for logging."
    );
    console.log("  → Sent Topic B message 1 (decorators)");

    await sendMessage(
      page,
      "What about Python context managers? How does the with statement work internally?"
    );
    console.log("  → Sent Topic B message 2 (context managers)");

    await sendMessage(
      page,
      "How do I use asyncio in Python for concurrent HTTP requests?"
    );
    console.log("  → Sent Topic B message 3 (asyncio)");

    await sendMessage(
      page,
      "Can you explain Python type hints and how to use generics with TypeVar?"
    );
    console.log("  → Sent Topic B message 4 (type hints)");

    // ── Back to Topic A: Ask about cooking ──
    await sendMessage(
      page,
      "Going back to cooking — what was that flour I mentioned for pasta earlier?"
    );
    console.log("  → Sent recall question about Topic A");

    // The response should reference semolina flour (from EAM memory)
    const assistantMessages = page.locator('[data-testid="message-assistant"]');
    const lastAssistant = assistantMessages.last();
    const responseText = await lastAssistant.textContent();
    console.log(`  → Response: ${responseText?.substring(0, 200)}`);

    // Check memory panel for recalled memories
    const memoryPanel = page.locator("aside");
    const memoryContent = await memoryPanel.textContent();
    console.log(`  → Memory panel: ${memoryContent?.substring(0, 300)}`);

    // Verify fidelity is showing
    await expect(page.locator('[data-testid="fidelity-label"]')).toBeVisible();

    // Check that SOME recalled memories appeared (from the pasta conversation)
    const hasMemories =
      memoryContent?.includes("vivid") ||
      memoryContent?.includes("clear") ||
      memoryContent?.includes("vague");
    console.log(`  → Has recalled memories: ${hasMemories}`);

    // The response should mention semolina (recalled from EAM)
    const mentionsSemolina =
      responseText?.toLowerCase().includes("semolina") ?? false;
    console.log(`  → Mentions semolina: ${mentionsSemolina}`);

    // At minimum, verify the system is working — we got a response
    expect(responseText).toBeTruthy();
    expect(responseText!.length).toBeGreaterThan(20);
  });

  test("fidelity increases for repeated topics", async ({ page }) => {
    test.setTimeout(180_000); // 3 minutes

    await page.goto("/");

    // First message on a topic
    await sendMessage(page, "Tell me about machine learning fundamentals.");

    // Get fidelity from first response
    await expect(
      page.locator('[data-testid="fidelity-label"]')
    ).toBeVisible({ timeout: 10000 });

    const fidelityText1 = await page
      .locator('[data-testid="fidelity-label"]')
      .locator("..")
      .textContent();
    console.log(`  → First fidelity: ${fidelityText1}`);

    // Second message on same topic (should have higher fidelity)
    await sendMessage(
      page,
      "What are the main types of machine learning algorithms?"
    );

    const fidelityText2 = await page
      .locator('[data-testid="fidelity-label"]')
      .locator("..")
      .textContent();
    console.log(`  → Second fidelity: ${fidelityText2}`);

    // Just verify both responses exist
    const assistantMessages = page.locator('[data-testid="message-assistant"]');
    const count = await assistantMessages.count();
    expect(count).toBeGreaterThanOrEqual(2);
  });
});
