// Minimal markdown → HTML. All text is escaped before any tag is added, so
// nothing in an answer (which may quote a web page) can become markup.

export const escHtml = (s) => String(s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export function renderMd(text) {
  if (!text) return "";
  // Code is set aside first so its contents are not treated as markdown.
  const stash = [];
  const keep = (html) => `\u0000CODE${stash.push(html) - 1}\u0000`;
  text = text.replace(/\u0000/g, "");
  text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) =>
    keep(`<pre><code class="lang-${escHtml(lang)}">${escHtml(code.trimEnd())}</code></pre>`));
  text = text.replace(/`([^`\n]+)`/g, (_, c) => keep(`<code>${escHtml(c)}</code>`));
  text = escHtml(text);
  // Headings
  text = text.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  text = text.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  text = text.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  // Bold/italic
  text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^\s*][^*]*?)\*(?!\*)/g, "$1<em>$2</em>");
  // Links
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // Bare URLs
  text = text.replace(/(^|\s)(https?:\/\/[^\s<]+)/g,
    '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  // Blockquote (the ">" is escaped by now)
  text = text.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");
  // Lists (simple)
  text = text.replace(/^(?:[-*] .+(?:\n|$))+/gm, (m) => {
    const items = m.trim().split("\n").map(l => l.replace(/^[-*] /, ""));
    return "<ul>" + items.map(i => `<li>${i}</li>`).join("") + "</ul>";
  });
  text = text.replace(/^(?:\d+\. .+(?:\n|$))+/gm, (m) => {
    const items = m.trim().split("\n").map(l => l.replace(/^\d+\.\s+/, ""));
    return "<ol>" + items.map(i => `<li>${i}</li>`).join("") + "</ol>";
  });
  // Paragraphs (a code block on its own must not be wrapped in <p>)
  text = text.split(/\n{2,}/).map(p => {
    const t = p.trim();
    const code = /^\u0000CODE(\d+)\u0000$/.exec(t);
    if (code && stash[+code[1]]?.startsWith("<pre>")) return t;
    if (/^<(h\d|ul|ol|blockquote)/.test(t)) return p;
    return "<p>" + p.replace(/\n/g, "<br>") + "</p>";
  }).join("\n");
  return text.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => stash[+i] ?? "");
}
