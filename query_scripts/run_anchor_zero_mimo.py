#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_anchor_zero_mimo.py — 一审判赔=0(一审驳回、全程0)的案子单独跑。
对这些案:锚=0,扰动无意义(0×k=0),只跑【真锚×1(锚=0)】一条,代表其全部带锚条件。
官方 MiMo,mimo-v2.5-pro 非思考、T=0,带锚直问"判多少"。

输出:
  ../results/anchor_zero_mimo25pro.json   (这15个案的真锚×1=锚0 结果)
  ../15案_一审驳回0.json                  (15个案号清单,留底)
回车后弹出隐藏输入贴 key(不留存)。
"""
import os, re, json, sys, threading, getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE  = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RESDIR= os.path.join(ROOT, "results"); os.makedirs(RESDIR, exist_ok=True)
CASES = json.load(open(os.path.join(ROOT, '案件_165案_语义修复版.json')))['cases']
P     = json.load(open(os.path.join(ROOT, 'prompts.json')))

MODEL = os.environ.get("RUN_MODEL", "mimo-v2.5-pro")
BASE  = os.environ.get("RUN_BASE",  "https://api.xiaomimimo.com/v1")
KEY   = os.environ.get("RUN_KEY", "").strip()
TAG   = os.environ.get("RUN_TAG", "mimo25pro").strip()
PAR   = max(1, int(os.environ.get("RUN_PARALLEL", "30")))
EXTRA = json.loads(os.environ["RUN_EXTRA"]) if os.environ.get("RUN_EXTRA") else {"thinking": {"type": "disabled"}}   # RUN_EXTRA 覆盖(OpenRouter grok: '{"reasoning":{"enabled":false}}')
OUT   = os.path.join(RESDIR, f"anchor_zero_{TAG}.json")

AP = {'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉',
      '原审原告、原审被告上诉':'双方上诉'}

def is_zero(c):
    try: return float(c.get('一审实体赔偿金额')) == 0
    except: return False

def txt(f):
    if isinstance(f, dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')

def build(c):
    M = P['模板']['带锚二审']
    诉请=txt(c['一审原告诉请']); 查明=txt(c['一审认定事实']); 新增=txt(c['二审新增认定事实'])
    ap=AP.get(c.get('二审上诉方类型'), c.get('二审上诉方类型',''))
    return M['指令'], M['输入_带锚'].format(诉请=诉请, 一审查明=查明, 二审新增=新增, 锚=0, 上诉方=ap)

def parse_num(t):
    t = (t or '').strip().replace(',', '').replace('，', '')
    m = re.search(r'-?\d+\.?\d*', t)
    if m: return float(m.group())
    if re.search(r'true|驳回|不予支持|不承担|无需赔偿|应当?是?0', t, re.I): return 0.0
    return None

try:
    from openai import OpenAI
except ImportError:
    sys.exit("请先 pip install openai")
client = None; lock = threading.Lock()

def call(c):
    sysmsg, usr = build(c)
    r = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}], extra_body=EXTRA)
    raw = r.choices[0].message.content
    return {"a": c['CLI'], "cond": "真锚x1(锚0)", "锚": 0, "out": parse_num(raw),
            "raw": raw, "model": getattr(r, 'model', MODEL)}

if __name__ == "__main__":
    zeros = [c for c in CASES if is_zero(c)]
    # 留底:15个案号清单
    json.dump([c['CLI'] for c in zeros], open(os.path.join(ROOT, '15案_一审驳回0.json'),'w'), ensure_ascii=False, indent=1)
    print(f"一审驳回(判赔0)案 {len(zeros)} 个,案号已存 ../15案_一审驳回0.json")
    if not KEY:
        KEY = getpass.getpass("请输入 MiMo API key(输入时不显示、运行结束不留存): ").strip()
    if not KEY:
        sys.exit("未输入 key,退出。")
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=180, max_retries=6)
    done = {r['a']: r for r in json.load(open(OUT)) if r.get('out') is not None} if os.path.exists(OUT) else {}
    todo = [c for c in zeros if c['CLI'] not in done]
    res = list(done.values())
    print(f"真锚×1(锚=0):已完成 {len(done)} / 待跑 {len(todo)} → {OUT}")
    def save():
        with lock: json.dump(res, open(OUT,'w'), ensure_ascii=False, indent=1)
    with ThreadPoolExecutor(max_workers=PAR) as ex:
        futs = {ex.submit(call, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                rec = f.result(); res.append(rec)
                print(f"[{i}/{len(todo)}] {rec['a']} 锚0 → {rec.get('out') if rec.get('out') is not None else '⚠️'+repr((rec.get('raw') or '')[:30])}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {c['CLI']} → ERR {repr(e)}", flush=True)
    save()
    z = sum(1 for r in res if r.get('out') == 0); pos = sum(1 for r in res if r.get('out') not in (0, None))
    print(f"完成 {len(res)}:判0 {z} / 判正数 {pos} → {OUT}")
