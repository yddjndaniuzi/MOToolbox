// Shared lightweight UI helpers: toasts, themed confirm dialog, button loading.
// Exposed on window.MOUI so inline page scripts can use them without imports.
(function () {
  function ensureToastHost() {
    let host = document.getElementById("mo-toast-host");
    if (!host) {
      host = document.createElement("div");
      host.id = "mo-toast-host";
      host.className = "mo-toast-host";
      host.setAttribute("aria-live", "polite");
      host.setAttribute("aria-atomic", "false");
      document.body.appendChild(host);
    }
    return host;
  }

  // type: "info" | "success" | "error"
  function toast(message, type = "info", timeout = 3200) {
    const host = ensureToastHost();
    const node = document.createElement("div");
    node.className = `mo-toast mo-toast-${type}`;
    node.setAttribute("role", type === "error" ? "alert" : "status");
    node.textContent = message;
    host.appendChild(node);
    // Trigger enter transition on next frame.
    requestAnimationFrame(() => node.classList.add("is-visible"));
    const remove = () => {
      node.classList.remove("is-visible");
      node.addEventListener("transitionend", () => node.remove(), { once: true });
      window.setTimeout(() => node.remove(), 400);
    };
    if (timeout > 0) window.setTimeout(remove, timeout);
    node.addEventListener("click", remove);
    return remove;
  }

  // Themed replacement for window.confirm. Returns a Promise<boolean>.
  function confirmDialog(message, opts = {}) {
    const {
      title = "确认操作",
      confirmText = "确认",
      cancelText = "取消",
      danger = false,
    } = opts;
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "mo-modal-overlay";
      overlay.innerHTML = `
        <div class="mo-modal" role="dialog" aria-modal="true" aria-labelledby="mo-modal-title">
          <h3 id="mo-modal-title">${escapeHtml(title)}</h3>
          <p>${escapeHtml(message)}</p>
          <div class="mo-modal-actions">
            <button type="button" class="mo-modal-cancel">${escapeHtml(cancelText)}</button>
            <button type="button" class="mo-modal-confirm${danger ? " is-danger" : ""}">${escapeHtml(confirmText)}</button>
          </div>
        </div>`;
      document.body.appendChild(overlay);

      const cleanup = (result) => {
        document.removeEventListener("keydown", onKey);
        overlay.classList.remove("is-visible");
        window.setTimeout(() => overlay.remove(), 200);
        resolve(result);
      };
      const onKey = (event) => {
        if (event.key === "Escape") cleanup(false);
        if (event.key === "Enter") cleanup(true);
      };

      overlay.querySelector(".mo-modal-cancel").addEventListener("click", () => cleanup(false));
      overlay.querySelector(".mo-modal-confirm").addEventListener("click", () => cleanup(true));
      overlay.addEventListener("click", (event) => {
        if (event.target === overlay) cleanup(false);
      });
      document.addEventListener("keydown", onKey);

      requestAnimationFrame(() => overlay.classList.add("is-visible"));
      overlay.querySelector(".mo-modal-confirm").focus();
    });
  }

  // Put a button into a loading state; returns a function to restore it.
  function setButtonLoading(button, loadingText) {
    if (!button) return () => {};
    const original = button.innerHTML;
    const wasDisabled = button.disabled;
    button.disabled = true;
    button.dataset.loading = "true";
    button.innerHTML = `<span class="mo-btn-spinner" aria-hidden="true"></span>${
      loadingText ? `<span>${escapeHtml(loadingText)}</span>` : ""
    }`;
    return () => {
      button.disabled = wasDisabled;
      delete button.dataset.loading;
      button.innerHTML = original;
    };
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  window.MOUI = { toast, confirm: confirmDialog, setButtonLoading };
})();
