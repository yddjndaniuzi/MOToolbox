document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-open-folder]");
  if (!button || button.disabled) return;
  button.disabled = true;
  try {
    const response = await fetch("/api/storage/open", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({location: button.dataset.openFolder, slug: button.dataset.folderSlug})
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "无法打开访达");
  } catch (error) {
    if (window.MOUI) window.MOUI.toast(error.message, "error");
    else window.alert(error.message);
  } finally {
    button.disabled = false;
  }
});
