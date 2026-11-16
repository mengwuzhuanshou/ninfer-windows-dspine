import json, sys, urllib.request

LABEL = sys.argv[1]
CASES = [("counting", "dspine-smoke-req2.json"), ("code", "req_code.json"),
         ("list", "req_list.json"), ("prose", "req_prose.json")]
rows = []
for name, path in CASES:
    body = open(path, "rb").read()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    open(f"dspine_ab_{LABEL}_{name}.json", "w", encoding="utf-8").write(
        json.dumps(d, ensure_ascii=False, indent=1))
    t = d["timings"]
    acc = t["draft_n_accepted"] / t["draft_n"] * 100 if t["draft_n"] else 0.0
    text = d["choices"][0]["message"]["content"]
    rows.append((name, acc, t["predicted_per_second"], t["predicted_n"],
                 t["draft_n"], t["draft_n_accepted"], text))
    print(f"{name:9s} acc={acc:5.1f}%  tok/s={t['predicted_per_second']:6.1f}  "
          f"n={t['predicted_n']:4d}  draft={t['draft_n_accepted']}/{t['draft_n']}")
print("=== outputs ===")
for name, acc, tps, n, dn, da, text in rows:
    print(f"--- {name} ---\n{text}\n")