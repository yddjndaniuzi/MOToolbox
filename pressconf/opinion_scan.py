from __future__ import annotations

import csv
import html
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pressconf.config_store import resolve_model
from pressconf.derivatives import call_writing_model_streaming
from pressconf.runtime import certifi_ssl_context


SCAN_TEMPLATES: dict[str, dict[str, Any]] = {
    "product_launch": {
        "name": "本品上市舆情",
        "summary": "上市前后口碑、风险主题和代表性原文扫描。",
        "sources": ["微博", "哔哩哔哩", "小红书", "酷安"],
        "topics": ["价格", "影像", "续航", "发热", "系统", "屏幕", "做工", "渠道", "售后"],
        "risk_words": [
            "翻车",
            "投诉",
            "维权",
            "退款",
            "退货",
            "发热",
            "断流",
            "绿屏",
            "虚标",
            "偷工减料",
            "售后",
            "卡顿",
            "bug",
            "崩溃",
            "差评",
        ],
    },
    "overseas_competitor": {
        "name": "海外竞品发布",
        "summary": "海外平台事实收集、媒体评价、用户争议和对我方警示。",
        "sources": ["X", "Reddit", "YouTube", "Hacker News", "科技媒体"],
        "topics": ["feature", "launch", "hands-on", "review", "benchmark", "price", "camera", "battery", "AI"],
        "risk_words": [
            "issue",
            "bug",
            "controversy",
            "overheat",
            "expensive",
            "disappointing",
            "privacy",
            "delay",
            "lawsuit",
        ],
    },
    "llm_tech": {
        "name": "大模型技术舆情",
        "summary": "极客平台里模型、项目、框架的热度、开发者反馈和争议扫描。",
        "sources": ["X", "GitHub", "Hacker News", "Reddit", "技术博客"],
        "topics": ["benchmark", "agent", "RAG", "inference", "latency", "pricing", "open source", "license", "safety"],
        "risk_words": [
            "hallucination",
            "regression",
            "license",
            "copyright",
            "safety",
            "benchmark",
            "fake",
            "latency",
            "cost",
            "outage",
        ],
    },
}

SOURCE_SEARCH_SITES: dict[str, list[str]] = {
    "微博": ["weibo.com"],
    "哔哩哔哩": ["bilibili.com"],
    "小红书": ["xiaohongshu.com"],
    "酷安": ["coolapk.com"],
    "X": ["x.com", "twitter.com"],
    "GitHub": ["github.com"],
    "Hacker News": ["news.ycombinator.com", "hn.algolia.com"],
    "Reddit": ["reddit.com"],
    "YouTube": ["youtube.com"],
    "科技媒体": ["theverge.com", "9to5mac.com", "androidauthority.com", "techcrunch.com"],
    "技术博客": ["huggingface.co", "paperswithcode.com", "arxiv.org", "semianalysis.com"],
}

SOURCE_COLLECTION_STRATEGIES: dict[str, list[str]] = {
    "微博": [
        "优先使用平台内搜索或已有舆情平台导出结果，再导入链接/CSV；通用搜索引擎覆盖很弱。",
        "可做浏览器辅助采集：用本机已登录浏览器打开搜索结果页，由用户控制翻页和筛选，只读取当前可见公开内容。",
        "如要稳定规模化，建议评估合规数据服务或商业舆情接口。",
    ],
    "哔哩哔哩": [
        "公开视频搜索、UP 主主页和视频评论可以作为半自动种子，但评论区需要控制频率和采集范围。",
        "MVP 阶段优先导入视频链接、标题、简介、评论精选或第三方导出 CSV。",
    ],
    "小红书": [
        "不建议第一版硬爬；平台反爬和合规成本高，通用搜索引擎基本不可依赖。",
        "短期建议走新榜等数据平台、内部已有舆情工具导出、人工粘贴笔记链接/正文摘要。",
        "浏览器辅助只适合小样本工作流：用户登录后手动搜索和筛选，工具读取用户确认的当前可见内容。",
    ],
    "酷安": [
        "可优先做公开页面和站内搜索的小样本采集评估；手机产品讨论密度高，值得作为半自动源优先试。",
        "稳定版本仍建议支持 CSV/链接导入，避免把报告质量绑死在页面结构上。",
    ],
    "X": [
        "优先用官方 API、授权第三方搜索服务或本机浏览器辅助小样本采集。",
        "如使用浏览器登录态，应限定为用户主动打开的搜索页和当前可见公开内容。",
    ],
    "GitHub": [
        "适合直接接 GitHub API：repo、issue、PR、discussion、release、stars 等结构化数据。",
    ],
    "Hacker News": [
        "适合接 HN 官方/Algolia API，稳定性和可追溯性都比较好。",
    ],
    "Reddit": [
        "优先使用 Reddit API 或授权第三方数据服务；按 subreddit 和关键词扫描。",
    ],
    "YouTube": [
        "适合接 YouTube Data API 或导入视频链接；评论采集需要按配额和合规边界做。",
    ],
    "科技媒体": [
        "优先接 RSS、站点 sitemap、新闻 API 或人工维护媒体白名单。",
    ],
    "技术博客": [
        "优先接 RSS、GitHub/论文索引/API 和手动链接导入。",
    ],
}

POSITIVE_WORDS = [
    "好评",
    "惊喜",
    "推荐",
    "优秀",
    "领先",
    "稳定",
    "流畅",
    "喜欢",
    "满意",
    "impressive",
    "great",
    "love",
    "solid",
    "useful",
    "fast",
]
NEGATIVE_WORDS = [
    "差评",
    "失望",
    "离谱",
    "翻车",
    "投诉",
    "难用",
    "不值",
    "bug",
    "disappointing",
    "bad",
    "broken",
    "issue",
    "problem",
    "expensive",
]

BROWSER_CAPTURE_BOOKMARKLET = (
    "javascript:(async()=>{"
    "const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
    "const visible=e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0&&r.bottom>0&&r.right>0&&r.top<innerHeight&&r.left<innerWidth};"
    "const links=[...document.querySelectorAll('a[href]')].filter(visible).map(a=>({text:clean(a.innerText||a.textContent||a.title),href:a.href})).filter(x=>x.text&&x.href).slice(0,120);"
    "const payload={title:document.title,url:location.href,captured_at:new Date().toISOString(),selected_text:clean(String(getSelection&&getSelection()||'')),text:clean(document.body&&document.body.innerText||'').slice(0,20000),links};"
    "try{"
    "const r=await fetch('http://127.0.0.1:5058/api/opinion-scan/browser-capture',{method:'POST',mode:'cors',headers:{'Content-Type':'text/plain'},body:JSON.stringify(payload)});"
    "alert(r.ok?'已发送到 MO 舆情扫描':'发送失败：'+r.status);"
    "}catch(e){alert('发送失败，请确认 MO 工具箱 5058 服务已打开：'+e.message)}"
    "})()"
)


@dataclass
class OpinionItem:
    title: str
    url: str = ""
    source: str = ""
    author: str = ""
    published_at: str = ""
    content: str = ""
    relevance: int = 0
    sentiment: str = "中性/事实"
    risk_score: int = 0
    topics: list[str] | None = None
    matched_terms: list[str] | None = None
    summary: str = ""


class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self._in_title = False
        self._skip = False
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attrs_dict = {key.lower(): value or "" for key, value in attrs}
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self.meta_description = attrs_dict.get("content", "")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = normalize_space(data)
        if not text:
            return
        if self._in_title:
            self.title = normalize_space(f"{self.title} {text}")
        elif not self._skip and len(text) >= 8:
            self._chunks.append(text)

    def body_text(self) -> str:
        return normalize_space(" ".join(self._chunks[:140]))


def template_defaults(template_key: str) -> dict[str, Any]:
    template = SCAN_TEMPLATES.get(template_key) or SCAN_TEMPLATES["product_launch"]
    return {
        "template_key": template_key if template_key in SCAN_TEMPLATES else "product_launch",
        "template": template,
        "sources": template["sources"],
        "risk_words": "\n".join(template["risk_words"]),
        "topics": "\n".join(template["topics"]),
    }


def save_browser_capture(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    capture = normalize_browser_capture(payload)
    inbox_path = root / "browser_captures.jsonl"
    with inbox_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(capture, ensure_ascii=False) + "\n")
    return capture


def capture_active_browser_tab(browser: str = "chrome") -> dict[str, Any]:
    browser = (browser or "chrome").strip().lower()
    app_name = {
        "chrome": "Google Chrome",
        "safari": "Safari",
    }.get(browser)
    if not app_name:
        raise ValueError("暂只支持 Chrome 或 Safari。")
    script = active_tab_applescript(app_name)
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(browser_capture_error(browser, message)) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("读取当前标签页超时，请确认浏览器当前页已经加载完成。") from exc
    output = (result.stdout or "").strip()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("浏览器返回内容不是有效 JSON，可能是当前页面禁止脚本读取。") from exc
    return normalize_browser_capture(payload)


def active_tab_applescript(app_name: str) -> str:
    js = browser_capture_javascript()
    escaped_js = js.replace("\\", "\\\\").replace('"', '\\"')
    if app_name == "Safari":
        return f'''
tell application "Safari"
    if not (exists front window) then error "Safari 没有打开窗口"
    set payload to do JavaScript "{escaped_js}" in current tab of front window
end tell
return payload
'''.strip()
    return f'''
tell application "{app_name}"
    if not (exists front window) then error "{app_name} 没有打开窗口"
    set payload to execute active tab of front window javascript "{escaped_js}"
end tell
return payload
'''.strip()


def browser_capture_javascript() -> str:
    return r"""(() => {
const clean = (s) => String(s || '').replace(/\s+/g, ' ').trim();
const visible = (e) => {
  const r = e.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
};
const links = Array.from(document.querySelectorAll('a[href]'))
  .filter(visible)
  .map((a) => ({ text: clean(a.innerText || a.textContent || a.title), href: a.href }))
  .filter((x) => x.text && x.href)
  .slice(0, 160);
return JSON.stringify({
  title: document.title,
  url: location.href,
  captured_at: new Date().toISOString(),
  selected_text: clean(String(getSelection && getSelection() || '')),
  text: clean(document.body && document.body.innerText || '').slice(0, 30000),
  links
});
})()"""


def browser_capture_error(browser: str, message: str) -> str:
    if browser == "chrome":
        return (
            "无法读取 Chrome 当前标签页。请确认 Chrome 已打开目标页面，并在 Chrome 菜单开启 "
            "View > Developer > Allow JavaScript from Apple Events。"
            + (f" 原始错误：{message}" if message else "")
        )
    return (
        "无法读取 Safari 当前标签页。请确认 Safari 已打开目标页面，并允许自动化控制 Safari。"
        + (f" 原始错误：{message}" if message else "")
    )


def load_browser_captures(root: Path, limit: int = 20) -> list[dict[str, Any]]:
    inbox_path = root / "browser_captures.jsonl"
    if not inbox_path.exists():
        return []
    lines = inbox_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    captures: list[dict[str, Any]] = []
    for line in reversed(lines[-200:]):
        try:
            captures.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(captures) >= limit:
            break
    return captures


def format_browser_captures(captures: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for capture in captures:
        title = normalize_space(capture.get("title", "")) or "浏览器采集页面"
        url = normalize_space(capture.get("url", ""))
        source = infer_source(url) or "浏览器采集"
        text = normalize_space(capture.get("selected_text") or capture.get("text") or "")
        links = capture.get("links") or []
        link_lines = []
        for link in links[:30]:
            label = normalize_space(link.get("text", ""))[:140]
            href = normalize_space(link.get("href", ""))
            if label and href:
                link_lines.append(f"- {label}\n{href}")
        content = "\n".join(
            part
            for part in [
                f"[{source}] {title}",
                url,
                text[:2500],
                "\n".join(link_lines),
            ]
            if part
        )
        if content:
            blocks.append(content)
    return "\n\n".join(blocks)


def normalize_browser_capture(payload: dict[str, Any]) -> dict[str, Any]:
    url = normalize_space(payload.get("url", ""))
    links = []
    for link in payload.get("links") or []:
        href = normalize_space((link or {}).get("href", ""))
        text = normalize_space((link or {}).get("text", ""))
        if href and text:
            links.append({"text": text[:240], "href": href[:1000]})
        if len(links) >= 120:
            break
    return {
        "title": normalize_space(payload.get("title", ""))[:240],
        "url": url[:1000],
        "source": infer_source(url) or "浏览器采集",
        "captured_at": normalize_space(payload.get("captured_at", ""))[:80],
        "selected_text": normalize_space(payload.get("selected_text", ""))[:8000],
        "text": normalize_space(payload.get("text", ""))[:20000],
        "links": links,
    }


def build_collection_guidance(target: str, keywords: list[str], sources: list[str], since: str = "") -> list[dict[str, Any]]:
    terms = [target, *keywords]
    query_terms = " ".join([term for term in terms if term]).strip()
    guidance: list[dict[str, Any]] = []
    for source in sources:
        direct_queries = build_direct_queries(source, query_terms, since)
        guidance.append(
            {
                "source": source,
                "queries": direct_queries,
                "strategies": SOURCE_COLLECTION_STRATEGIES.get(source, ["优先导入链接、CSV 或人工整理的文本，再进入统一分析流程。"]),
            }
        )
    return guidance


def build_direct_queries(source: str, query_terms: str, since: str = "") -> list[dict[str, str]]:
    query = f"{query_terms} {since}".strip()
    encoded = urllib.parse.quote(query)
    if not query:
        return []
    if source == "微博":
        return [{"label": "微博站内搜索", "url": f"https://s.weibo.com/weibo?q={encoded}"}]
    if source == "哔哩哔哩":
        return [{"label": "B站站内搜索", "url": f"https://search.bilibili.com/all?keyword={encoded}"}]
    if source == "小红书":
        return [{"label": "小红书站内搜索", "url": f"https://www.xiaohongshu.com/search_result?keyword={encoded}"}]
    if source == "酷安":
        return [{"label": "酷安站内搜索", "url": f"https://www.coolapk.com/search?q={encoded}"}]
    if source == "GitHub":
        return [{"label": "GitHub Search", "url": f"https://github.com/search?q={encoded}&type=repositories"}]
    if source == "Hacker News":
        return [{"label": "HN Algolia", "url": f"https://hn.algolia.com/?q={encoded}"}]
    if source == "Reddit":
        return [{"label": "Reddit Search", "url": f"https://www.reddit.com/search/?q={encoded}"}]
    if source == "YouTube":
        return [{"label": "YouTube Search", "url": f"https://www.youtube.com/results?search_query={encoded}"}]
    if source == "X":
        return [{"label": "X Search", "url": f"https://x.com/search?q={encoded}&src=typed_query&f=live"}]
    return []


def run_scan(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = normalize_space(payload.get("target", ""))
    template_key = payload.get("template_key", "product_launch")
    template = SCAN_TEMPLATES.get(template_key) or SCAN_TEMPLATES["product_launch"]
    keywords = split_terms(payload.get("keywords", ""))
    aliases = split_terms(payload.get("aliases", ""))
    excludes = split_terms(payload.get("excludes", ""))
    risk_words = split_terms(payload.get("risk_words", "")) or list(template["risk_words"])
    topics = split_terms(payload.get("topics", "")) or list(template["topics"])
    sources = payload.get("sources") or list(template["sources"])
    items = collect_items(payload, sources)
    if payload.get("fetch_urls"):
        items = enrich_url_items(items)
    deduped = dedupe_items(items)
    analyzed = [
        analyze_item(item, target=target, keywords=keywords, aliases=aliases, excludes=excludes, risk_words=risk_words, topics=topics)
        for item in deduped
    ]
    relevant = [item for item in analyzed if item.relevance > 0]
    report = build_report(
        target=target,
        template_key=template_key,
        template=template,
        sources=sources,
        keywords=keywords,
        aliases=aliases,
        risk_words=risk_words,
        topics=topics,
        items=relevant,
        all_items=analyzed,
        collection_guidance=build_collection_guidance(target, keywords + aliases, sources, payload.get("time_range", "")),
    )
    if payload.get("use_ai"):
        report = ai_polish_report(report, relevant, payload, output_dir)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "items.json").write_text(json.dumps([asdict(item) for item in analyzed], ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "target": target,
        "template_key": template_key,
        "template_name": template["name"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_items": len(analyzed),
        "relevant_items": len(relevant),
        "high_risk_items": len([item for item in relevant if item.risk_score >= 3]),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta": meta, "items": analyzed, "report": report}


def collect_items(payload: dict[str, Any], sources: list[str]) -> list[OpinionItem]:
    items: list[OpinionItem] = []
    pasted = str(payload.get("pasted_content") or "").strip()
    if pasted:
        items.extend(parse_pasted_items(pasted, sources[0] if sources else "手动导入"))
    for row in payload.get("csv_rows") or []:
        item = item_from_row(row)
        if item:
            items.append(item)
    direct_urls = split_lines(payload.get("urls", ""))
    items.extend(OpinionItem(title=url, url=url, source=infer_source(url) or "链接导入") for url in direct_urls)
    return items


def parse_pasted_items(text: str, default_source: str) -> list[OpinionItem]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    items: list[OpinionItem] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        url = next((line for line in lines if line.startswith(("http://", "https://"))), "")
        title = lines[0] if not lines[0].startswith(("http://", "https://")) else url
        content = "\n".join(line for line in lines[1:] if line != url)
        items.append(OpinionItem(title=title[:180], url=url, source=infer_source(url) or default_source, content=content[:4000]))
    return items


def item_from_row(row: dict[str, str]) -> OpinionItem | None:
    title = first_value(row, ["title", "标题", "内容", "正文", "text", "content"])
    url = first_value(row, ["url", "链接", "原文链接", "link"])
    content = first_value(row, ["content", "正文", "内容", "摘要", "summary", "text"])
    if not title and not content and not url:
        return None
    return OpinionItem(
        title=(title or content or url)[:180],
        url=url,
        source=first_value(row, ["source", "来源", "平台", "platform"]) or infer_source(url) or "CSV",
        author=first_value(row, ["author", "作者", "账号", "user"]),
        published_at=first_value(row, ["published_at", "发布时间", "time", "date"]),
        content=content[:4000],
    )


def enrich_url_items(items: list[OpinionItem]) -> list[OpinionItem]:
    enriched = []
    for item in items:
        if item.url and (not item.content or item.title == item.url):
            fetched = fetch_url_text(item.url)
            if fetched:
                item.title = fetched.get("title") or item.title
                item.content = fetched.get("content") or item.content
                item.source = item.source or infer_source(item.url) or "网页"
        enriched.append(item)
    return enriched


def fetch_url_text(url: str) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 MOtoolbox opinion scan preview"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=12, context=certifi_ssl_context()) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return {}
            raw = response.read(800_000)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {}
    text = raw.decode("utf-8", errors="ignore")
    parser = PageTextParser()
    parser.feed(text)
    title = html.unescape(parser.title).strip()
    description = html.unescape(parser.meta_description).strip()
    body = parser.body_text()
    return {"title": title or description or url, "content": normalize_space(f"{description} {body}")[:4000]}


def dedupe_items(items: list[OpinionItem]) -> list[OpinionItem]:
    seen: set[str] = set()
    result: list[OpinionItem] = []
    for item in items:
        key = normalize_dedupe_key(item.url or item.title or item.content)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def analyze_item(
    item: OpinionItem,
    *,
    target: str,
    keywords: list[str],
    aliases: list[str],
    excludes: list[str],
    risk_words: list[str],
    topics: list[str],
) -> OpinionItem:
    text = f"{item.title}\n{item.content}".lower()
    terms = [target, *keywords, *aliases]
    matched = [term for term in terms if term and term.lower() in text]
    excluded = [term for term in excludes if term and term.lower() in text]
    item.relevance = max(0, len(matched) * 2 - len(excluded) * 3)
    item.matched_terms = matched
    matched_risks = [word for word in risk_words if word and word.lower() in text]
    negative_hits = [word for word in NEGATIVE_WORDS if word.lower() in text]
    positive_hits = [word for word in POSITIVE_WORDS if word.lower() in text]
    item.risk_score = len(matched_risks) * 2 + len(negative_hits)
    if item.risk_score >= 3:
        item.sentiment = "负面/风险"
    elif positive_hits and not negative_hits:
        item.sentiment = "正面"
    elif positive_hits and negative_hits:
        item.sentiment = "混合"
    else:
        item.sentiment = "中性/事实"
    item.topics = [topic for topic in topics if topic and topic.lower() in text][:6]
    item.summary = summarize_item(item, matched_risks, positive_hits, negative_hits)
    return item


def summarize_item(item: OpinionItem, risks: list[str], positives: list[str], negatives: list[str]) -> str:
    source_text = normalize_space(item.content or item.title)
    snippet = source_text[:130] + ("..." if len(source_text) > 130 else "")
    signals = []
    if risks:
        signals.append("风险词：" + "、".join(risks[:4]))
    if positives:
        signals.append("正向词：" + "、".join(positives[:3]))
    if negatives:
        signals.append("负向词：" + "、".join(negatives[:3]))
    if signals:
        return f"{snippet}（{'；'.join(signals)}）"
    return snippet


def build_report(
    *,
    target: str,
    template_key: str,
    template: dict[str, Any],
    sources: list[str],
    keywords: list[str],
    aliases: list[str],
    risk_words: list[str],
    topics: list[str],
    items: list[OpinionItem],
    all_items: list[OpinionItem],
    collection_guidance: list[dict[str, Any]],
) -> str:
    high_risk = sorted([item for item in items if item.risk_score >= 3], key=lambda item: item.risk_score, reverse=True)
    topic_counts = count_topics(items)
    sentiment_counts = count_by(items, "sentiment")
    source_counts = count_by(items, "source")
    lines = [
        f"# {target or '未命名对象'} - {template['name']}扫描报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 模板：{template['name']}",
        f"- 数据来源：{', '.join(sources) if sources else '手动导入'}",
        f"- 输入内容：{len(all_items)} 条；相关内容：{len(items)} 条；高风险内容：{len(high_risk)} 条",
        "",
        "## 结论摘要",
        "",
        conclusion_summary(template_key, items, high_risk, sentiment_counts, topic_counts),
        "",
        "## 关键发现",
        "",
    ]
    lines.extend(key_findings(template_key, high_risk, topic_counts, sentiment_counts))
    lines.extend(["", "## 话题分布", ""])
    lines.extend(markdown_count_table(topic_counts, "话题"))
    lines.extend(["", "## 情绪/态度分布", ""])
    lines.extend(markdown_count_table(sentiment_counts, "态度"))
    lines.extend(["", "## 来源分布", ""])
    lines.extend(markdown_count_table(source_counts, "来源"))
    lines.extend(["", "## 高优先级内容", ""])
    lines.extend(item_table(high_risk[:12]))
    lines.extend(["", "## 代表性内容", ""])
    lines.extend(item_table(sorted(items, key=lambda item: (item.risk_score, item.relevance), reverse=True)[:20]))
    lines.extend(["", "## 建议动作", ""])
    lines.extend(recommendations(template_key, high_risk, topic_counts))
    lines.extend(["", "## 数据采集建议", ""])
    if collection_guidance:
        for item in collection_guidance:
            lines.append(f"### {item['source']}")
            queries = item.get("queries") or []
            if queries:
                for query in queries:
                    lines.append(f"- [{query['label']}]({query['url']})")
            for strategy in item.get("strategies") or []:
                lines.append(f"- {strategy}")
            lines.append("")
    else:
        lines.append("- 暂无。")
    lines.extend(["", "## 扫描口径", ""])
    lines.append(f"- 关键词：{', '.join([target, *keywords, *aliases])}")
    lines.append(f"- 风险词：{', '.join(risk_words[:40])}")
    lines.append(f"- 主题词：{', '.join(topics[:40])}")
    return "\n".join(lines).strip() + "\n"


def ai_polish_report(report: str, items: list[OpinionItem], payload: dict[str, Any], output_dir: Path) -> str:
    model_config = resolve_model(Path(payload.get("base_dir") or output_dir), "writing", "standard")
    if not model_config.get("api_key"):
        return report + "\n\n> AI 聚合未运行：后台模型 API Key 未配置。\n"
    compact_items = "\n".join(
        f"- [{item.source}] {item.title} | {item.sentiment} | 风险 {item.risk_score} | {item.summary} | {item.url}"
        for item in items[:60]
    )
    prompt = f"""
请基于以下规则分析结果，重写为一份更适合市场/产品团队内部同步的中文扫描报告。

要求：
1. 不要编造输入里没有的事实、平台、账号、链接、比例或结论。
2. 保留可追溯链接。
3. 区分“已经发生的事实”和“需要人工复核的推断”。
4. 输出 Markdown。

原始报告：
{report}

结构化条目：
{compact_items}
""".strip()
    try:
        ai_report = call_writing_model_streaming(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            model=model_config["model"],
            provider=model_config["provider"],
            prompt=prompt,
            max_tokens=9000,
            stage="opinion_scan",
            system_prompt="你是资深舆情和竞品情报分析师，擅长输出克制、可追溯、可行动的中文扫描报告。",
            fallback_models=model_config.get("fallbacks"),
        )
    except Exception as exc:
        return report + f"\n\n> AI 聚合未完成：{exc}\n"
    return ai_report.strip() + "\n"


def read_csv_upload(file_storage: Any) -> list[dict[str, str]]:
    if not file_storage or not getattr(file_storage, "filename", ""):
        return []
    raw = file_storage.read()
    text = raw.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(text.splitlines())
    return [{str(key or ""): str(value or "") for key, value in row.items()} for row in reader]


def count_topics(items: list[OpinionItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for topic in item.topics or ["未归类"]:
            counts[topic] = counts.get(topic, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: pair[1], reverse=True))


def count_by(items: list[OpinionItem], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(getattr(item, attr) or "未标注")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: pair[1], reverse=True))


def conclusion_summary(
    template_key: str,
    items: list[OpinionItem],
    high_risk: list[OpinionItem],
    sentiment_counts: dict[str, int],
    topic_counts: dict[str, int],
) -> str:
    top_topics = "、".join(list(topic_counts)[:3]) or "暂无明显聚类"
    negative = sentiment_counts.get("负面/风险", 0)
    if not items:
        return "本轮输入中没有识别到足够相关内容，建议补充链接、CSV 或扩大关键词后重新扫描。"
    if template_key == "product_launch":
        return f"本轮识别到 {len(items)} 条相关内容，其中负面/风险 {negative} 条，高风险 {len(high_risk)} 条；讨论主要集中在 {top_topics}。"
    if template_key == "overseas_competitor":
        return f"本轮识别到 {len(items)} 条竞品相关内容，主要信息集中在 {top_topics}；其中 {len(high_risk)} 条需要作为争议或警示复核。"
    return f"本轮识别到 {len(items)} 条大模型相关内容，讨论集中在 {top_topics}；其中 {len(high_risk)} 条涉及技术争议、成本、许可或安全等风险信号。"


def key_findings(template_key: str, high_risk: list[OpinionItem], topic_counts: dict[str, int], sentiment_counts: dict[str, int]) -> list[str]:
    lines = []
    if topic_counts:
        lines.append(f"- 最高频话题是 **{next(iter(topic_counts))}**，建议优先人工复核相关原文。")
    if high_risk:
        lines.append(f"- 发现 **{len(high_risk)}** 条高风险内容，风险最高的条目来自 **{high_risk[0].source or '未知来源'}**。")
    negative = sentiment_counts.get("负面/风险", 0)
    if negative:
        lines.append(f"- 负面/风险内容共 **{negative}** 条，应结合链接原文判断是否已经形成扩散。")
    if not lines:
        lines.append("- 暂未发现明显高风险聚类，当前结果更适合作为事实和素材收集。")
    if template_key == "overseas_competitor":
        lines.append("- 竞品扫描建议重点复核“事实是否准确”和“是否可转化为我方发布/销售/产品警示”。")
    if template_key == "llm_tech":
        lines.append("- 技术舆情建议额外复核 GitHub issue、benchmark 口径和 license 约束。")
    return lines


def item_table(items: list[OpinionItem]) -> list[str]:
    if not items:
        return ["暂无。"]
    lines = ["| 来源 | 标题/内容 | 态度 | 风险 | 主题 | 链接 |", "| --- | --- | --- | ---: | --- | --- |"]
    for item in items:
        title = escape_table(item.title or item.summary or "未命名")[:120]
        link = f"[原文]({item.url})" if item.url else "-"
        lines.append(
            f"| {escape_table(item.source or '-')} | {title} | {escape_table(item.sentiment)} | {item.risk_score} | {escape_table('、'.join(item.topics or [])) or '-'} | {link} |"
        )
    return lines


def markdown_count_table(counts: dict[str, int], label: str) -> list[str]:
    if not counts:
        return ["暂无。"]
    lines = [f"| {label} | 数量 |", "| --- | ---: |"]
    for key, value in counts.items():
        lines.append(f"| {escape_table(key)} | {value} |")
    return lines


def recommendations(template_key: str, high_risk: list[OpinionItem], topic_counts: dict[str, int]) -> list[str]:
    if template_key == "product_launch":
        lines = [
            "- 对高风险原文逐条复核，确认是否为真实用户集中反馈、单点个案或转载扩散。",
            "- 若风险集中在同一产品体验点，建议同步产品、客服和传播口径负责人。",
        ]
    elif template_key == "overseas_competitor":
        lines = [
            "- 将发布事实与我方产品卖点表交叉核对，标注可借鉴、需防御和可反打的点。",
            "- 对争议点保留原文链接，避免把海外用户情绪误写成确定事实。",
        ]
    else:
        lines = [
            "- 对高热项目补充 GitHub stars、issue 和 license 复核，确认是否有落地价值。",
            "- 对 benchmark、安全、成本类争议保留原始讨论，避免只引用二手总结。",
        ]
    if high_risk:
        lines.append("- 高风险内容建议优先按“影响力、是否跨平台、是否有证据链”三项人工排序。")
    if topic_counts:
        lines.append(f"- 下一轮扫描可围绕“{next(iter(topic_counts))}”增加更细关键词。")
    return lines


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in re.split(r"[,，;\n]+", str(value or "")) if term.strip()]


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_dedupe_key(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[?#].*$", "", value)
    return value[:220]


def infer_source(url: str) -> str:
    host = urllib.parse.urlparse(url or "").netloc.lower()
    if "weibo" in host:
        return "微博"
    if "bilibili" in host:
        return "哔哩哔哩"
    if "xiaohongshu" in host:
        return "小红书"
    if "coolapk" in host:
        return "酷安"
    if "github" in host:
        return "GitHub"
    if "reddit" in host:
        return "Reddit"
    if "ycombinator" in host or "algolia" in host:
        return "Hacker News"
    if "youtube" in host or "youtu.be" in host:
        return "YouTube"
    if "x.com" in host or "twitter" in host:
        return "X"
    return ""


def first_value(row: dict[str, str], keys: list[str]) -> str:
    lowered = {str(key).strip().lower(): str(value or "").strip() for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value:
            return value
    return ""


def escape_table(value: str) -> str:
    return normalize_space(value).replace("|", "\\|")
