#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_anchor_mimo.py — 真锚×1 + 真扰动×{1/3,2/3,3/2,3}(锚=真实一审额×倍数)。
官方 MiMo(api.xiaomimimo.com),mimo-v2.5-pro 非思考、T=0,带锚直问"判多少"。
★ 5×165=825 全部混成一个大池子,单线程池并发(不分段、中间不空转),落一个大 JSON。增量可续。

用法:
  python3 run_anchor_mimo.py                 # 全部 5 条件 × 165 混跑
  RUN_LIMIT=3 python3 run_anchor_mimo.py     # 每条件前 3 个(冒烟,共 15)
  RUN_PARALLEL=50 python3 run_anchor_mimo.py # 默认就是 50
回车后弹出隐藏输入贴 key(不留存)。
结果存 ../results/anchor_all_mimo25pro.json,每条:{a,cond,锚,out,raw,model}
"""
import os, re, json, sys, threading, getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE  = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RESDIR= os.path.join(ROOT, "results"); os.makedirs(RESDIR, exist_ok=True)
CASES = json.load(open(os.path.join(ROOT, '案件_165案_语义修复版.json')))['cases']
P     = json.load(open(os.path.join(ROOT, 'prompts.json')))

MODEL = os.environ.get("RUN_MODEL", "deepseek-chat")
BASE  = os.environ.get("RUN_BASE",  "https://api.deepseek.com")
KEY   = os.environ.get("RUN_KEY", "").strip()
TAG   = os.environ.get("RUN_TAG", "flash").strip()
PAR   = max(1, int(os.environ.get("RUN_PARALLEL", "50")))
LIMIT = int(os.environ.get("RUN_LIMIT", "0"))
EXTRA = {"thinking": {"type": "disabled"}}   # 官方关思考
OUT   = os.path.join(RESDIR, f"anchor_all_{TAG}.json")

CONDS = {  # 条件 → 倍数
    "真锚x1": 1.0, "真扰动x1/3": 1/3.0, "真扰动x2/3": 2/3.0, "真扰动x3/2": 1.5, "真扰动x3": 3.0,
}
AP = {'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉',
      '原审原告、原审被告上诉':'双方上诉'}

def txt(f):
    if isinstance(f, dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')

def anchor_val(c, factor):
    try: base = float(c.get('一审实体赔偿金额'))
    except: return None
    if base < 0: return None
    v = round(base * factor, 2)
    return int(v) if v == int(v) else v

def build(cond, c):
    M = P['模板']['带锚二审']
    a = anchor_val(c, CONDS[cond])
    if a is None: return None
    诉请=txt(c['一审原告诉请']); 查明=txt(c['一审认定事实']); 新增=txt(c['二审新增认定事实'])
    ap=AP.get(c.get('二审上诉方类型'), c.get('二审上诉方类型',''))
    return M['指令'], M['输入_带锚'].format(诉请=诉请, 一审查明=查明, 二审新增=新增, 锚=a, 上诉方=ap), a

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

def call(cond, c):
    built = build(cond, c)
    if built is None:
        return {"a": c['CLI'], "cond": cond, "锚": None, "out": None, "raw": "(无效一审额,跳过)", "model": MODEL}
    sysmsg, usr, a = built
    r = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}], extra_body=EXTRA)
    raw = r.choices[0].message.content
    return {"a": c['CLI'], "cond": cond, "锚": a, "out": parse_num(raw),
            "raw": raw, "model": getattr(r, 'model', MODEL)}

if __name__ == "__main__":
    conds = sys.argv[1:] or list(CONDS.keys())
    if not KEY:
        KEY = getpass.getpass("请输入 DeepSeek API key(输入时不显示、运行结束不留存): ").strip()
    if not KEY:
        sys.exit("未输入 key,退出。")
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=180, max_retries=6)

    def _base(c):
        try: return float(c.get('一审实体赔偿金额'))
        except: return None
    base_pool = CASES[:LIMIT] if LIMIT > 0 else CASES
    pos  = [c for c in base_pool if (_base(c) or -1) > 0]   # 一审额>0:150,跑全部5条件
    zero = [c for c in base_pool if _base(c) == 0]          # 一审额=0:15,只跑真锚x1(锚0)
    done = {}
    if os.path.exists(OUT):
        done = {f"{r['cond']}|{r['a']}": r for r in json.load(open(OUT)) if r.get('out') is not None}
    res = list(done.values())
    tasks_all = [(cond, c) for cond in conds for c in pos]
    if "真锚x1" in conds:
        tasks_all += [("真锚x1", c) for c in zero]
    tasks = [(cond, c) for cond, c in tasks_all if f"{cond}|{c['CLI']}" not in done]
    print(f"模型={MODEL} 关思考 T=0 并发={PAR} | 正锚{len(pos)}×{len(conds)} + 零锚{len(zero)} = 大盘 {len(tasks_all)} | 已完成 {len(done)} / 待跑 {len(tasks)} → {OUT}")

    def save():
        with lock: json.dump(res, open(OUT,'w'), ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(max_workers=PAR) as ex:
        futs = {ex.submit(call, cond, c): (cond, c) for cond, c in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            cond, c = futs[f]
            try:
                rec = f.result(); res.append(rec)
                o = rec.get('out')
                shown = o if o is not None else ('⚠️' + repr((rec.get('raw') or '')[:30]))
                print(f"[{i}/{len(tasks)}] {cond} {rec['a']} 锚{rec.get('锚')} → {shown}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(tasks)}] {cond} {c['CLI']} → ERR {repr(e)}", flush=True)
            if i % 30 == 0: save()
    save()
    # 汇总
    from collections import Counter
    print(f"\n完成 {len(res)} 条 → {OUT}")
    for cond in conds:
        sub = [r for r in res if r['cond'] == cond]
        z = sum(1 for r in sub if r.get('out') == 0); pos = sum(1 for r in sub if r.get('out') not in (0, None)); bad = sum(1 for r in sub if r.get('out') is None)
        print(f"  {cond}: {len(sub)} | 0值 {z} / 正数 {pos} / 失败 {bad}")
