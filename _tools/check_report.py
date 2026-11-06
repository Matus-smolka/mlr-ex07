import re, html, sys

t = open("report.html", encoding="utf-8").read()
t = re.sub(r"data:image/png;base64,[^\"']+", "[PNG]", t)
txt = html.unescape(re.sub(r"<[^>]+>", " ", t))
txt = re.sub(r"\s+", " ", txt)

print(f"report.html text, {len(txt):,} chars (images stripped)\n")
for kw in ["Was Adam handicapped", "imbalance was worth", "six largest confusions",
           "Three independent runs", "immature granulocyte", "does not establish"]:
    i = txt.find(kw)
    print("=" * 4, kw, "->", "FOUND" if i >= 0 else "*** MISSING ***")
    if i >= 0:
        print("   ", txt[i:i + 460].strip(), "\n")
