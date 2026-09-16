(() => {
  const button = document.getElementById('ytdlp-update');
  const label = document.getElementById('ytdlp-status');
  let timer;
  async function refresh() {
    clearTimeout(timer);
    try {
      const response = await fetch('/api/ytdlp');
      if (!response.ok) throw new Error('读取状态失败，请刷新页面重试');
      const state = await response.json();
      document.getElementById('ytdlp-version').textContent = state.version;
      button.disabled = state.running || !state.supported;
      button.textContent = state.running ? '更新中…' : '更新至最新稳定版';
      label.textContent = [state.message, state.error, state.version_error].filter(Boolean).join(' ');
      if (!state.supported) label.textContent = '一键更新目前支持 macOS。';
      if (state.running) timer = setTimeout(refresh, 2000);
    } catch (error) {
      label.textContent = error.message;
      timer = setTimeout(refresh, 5000);
    }
  }
  button.addEventListener('click', async () => {
    button.disabled = true;
    label.textContent = '正在启动更新…';
    try {
      const response = await fetch('/api/ytdlp/update', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
      const data = await response.json();
      if (!response.ok && response.status !== 409) throw new Error(data.error || '启动更新失败');
      await refresh();
    } catch (error) {
      label.textContent = error.message;
      button.disabled = false;
    }
  });
  refresh();
})();
