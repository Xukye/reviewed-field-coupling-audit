#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — 合并 runner:盲一审 + 盲二审 + 真锚x1 + 真扰动{1/3,2/3,3/2,3}(正锚) + 真锚x1(锚0)(零锚)
★ 全部混成一个大池子,单线程池并发(默认 30),增量可续。
★ prompt/parse 逐字复刻 run_blind / run_anchor / run_anchor_zero,输出仍分流到 4 个标准文件:
    blind1_<TAG>.json  blind2_<TAG>.json  anchor_all_<TAG>.json  anchor_zero_<TAG>.json
  → 下游分析零改动。

用法(与其它 runner 同一套 env):
  RUN_MODEL=x-ai/grok-4.3-20260430 RUN_BASE=https://openrouter.ai/api/v1 \
  RUN_TAG=grok43 RUN_EXTRA='{"reasoning":{"enabled":false}}' RUN_PARALLEL=30 \
  python3 run_all.py
  RUN_LIMIT=3 python3 run_all.py     # 冒烟:每类前 3
key:未设 RUN_KEY 则运行时 getpass 提示,不落盘。
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
LIMIT = int(os.environ.get("RUN_LIMIT", "0"))
EXTRA = json.loads(os.environ["RUN_EXTRA"]) if os.environ.get("RUN_EXTRA") else {"thinking": {"type": "disabled"}}

CONDS = {"真锚x1": 1.0, "真扰动x1/3": 1/3.0, "真扰动x2/3": 2/3.0, "真扰动x3/2": 1.5, "真扰动x3": 3.0}
AP = {'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉',
      '原审原告、原审被告上诉':'双方上诉'}
# 每类 → 输出文件
FILE_BLIND = {"盲一审": "blind1", "盲二审": "blind2"}

def txt(f):
    if isinstance(f, dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')

def anchor_val(c, factor):
    try: base = float(c.get('一审实体赔偿金额'))
    except: return None
    if base <= 0: return None
    v = round(base * factor, 2)
    return int(v) if v == int(v) else v

def is_pos(c):
    try: return float(c.get('一审实体赔偿金额')) > 0
    except: return False
def is_zero(c):
    try: return float(c.get('一审实体赔偿金额')) == 0
    except: return False

# ---- prompt 构建(逐字复刻各原 runner)----
def build(kind, cond, c):
    诉请=txt(c['一审原告诉请']); 查明=txt(c['一审认定事实']); 新增=txt(c['二审新增认定事实'])
    ap=AP.get(c.get('二审上诉方类型'), c.get('二审上诉方类型',''))
    if kind == 'blind':
        M = P['模板']
        if cond == '盲一审':
            return M['盲一审']['指令'], M['盲一审']['输入'].format(诉请=诉请, 一审查明=查明), None
        return M['盲二审']['指令'], M['盲二审']['输入'].format(诉请=诉请, 一审查明=查明, 二审新增=新增, 上诉方=ap), None
    M = P['模板']['带锚二审']
    if kind == 'zero':
        return M['指令'], M['输入_带锚'].format(诉请=诉请, 一审查明=查明, 二审新增=新增, 锚=0, 上诉方=ap), 0
    a = anchor_val(c, CONDS[cond])
    if a is None: return None
    return M['指令'], M['输入_带锚'].format(诉请=诉请, 一审查明=查明, 二审新增=新增, 锚=a, 上诉方=ap), a

# ---- parse:blind 用 T/F 门;anchor/zero 用纯数字 ----
def parse_blind(t):
    s = (t or '').strip(); head = s[:1].upper()
    m = re.search(r'-?\d+\.?\d*', s.replace(',', '').replace('，', ''))
    num = float(m.group()) if m else None
    if head == 'T': return 0.0
    if head == 'F': return num
    if num is not None: return num
    if re.search(r'驳回|不予支持|不承担|无需赔偿', s, re.I): return 0.0
    return None
def parse_anchor(t):
    t = (t or '').strip().replace(',', '').replace('，', '')
    m = re.search(r'-?\d+\.?\d*', t)
    if m: return float(m.group())
    if re.search(r'true|驳回|不予支持|不承担|无需赔偿|应当?是?0', t, re.I): return 0.0
    return None

def outfile(kind, cond):
    if kind == 'blind': return os.path.join(RESDIR, f"{FILE_BLIND[cond]}_{TAG}.json")
    if kind == 'zero':  return os.path.join(RESDIR, f"anchor_zero_{TAG}.json")
    return os.path.join(RESDIR, f"anchor_all_{TAG}.json")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("请先 pip install openai")
client = None; lock = threading.Lock()

def call(task):
    kind, cond, c = task
    built = build(kind, cond, c)
    if built is None:  # 正锚扰动遇到无效锚(不该发生,防御)
        return {"a": c['CLI'], "cond": cond, "锚": None, "out": None, "raw": "(无效一审额,跳过)", "model": MODEL, "_kind": kind}
    sysmsg, usr, a = built
    r = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}], extra_body=EXTRA)
    raw = r.choices[0].message.content
    out = parse_blind(raw) if kind == 'blind' else parse_anchor(raw)
    rec = {"a": c['CLI'], "cond": cond, "out": out, "raw": raw, "model": getattr(r, 'model', MODEL), "_kind": kind}
    if kind != 'blind':
        rec["锚"] = a
    return rec

if __name__ == "__main__":
    if not KEY:
        KEY = getpass.getpass("请输入 API key(输入时不显示、运行结束不留存): ").strip()
    if not KEY:
        sys.exit("未输入 key,退出。")
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=180, max_retries=6)

    pool = CASES[:LIMIT] if LIMIT > 0 else CASES
    pos = [c for c in pool if is_pos(c)]; zero = [c for c in pool if is_zero(c)]
    # 组装大池子
    tasks = []
    for c in pool:
        tasks += [('blind','盲一审',c), ('blind','盲二审',c)]
    for cond in CONDS:
        for c in pos:
            tasks.append(('anchor', cond, c))
    for c in zero:
        tasks.append(('zero', '真锚x1(锚0)', c))
    # 增量续跑:读已有 4 文件,out!=None 视为完成;dedup key = (outfile, cond, a)
    done = {}
    for f in set(outfile(k, cnd) for k in ('blind','anchor','zero') for cnd in (['盲一审','盲二审'] if k=='blind' else ['x'])):
        if os.path.exists(f):
            for r in json.load(open(f)):
                if r.get('out') is not None:
                    done[(f, r['cond'], r['a'])] = r
    todo = [t for t in tasks if (outfile(t[0], t[1]), t[1], t[2]['CLI']) not in done]
    print(f"模型={MODEL} 端点={BASE} 关思考={EXTRA} T=0 并发={PAR} | 大池 {len(tasks)} | 已完成 {len(done)} / 待跑 {len(todo)}"
          + (f" [冒烟前{LIMIT}]" if LIMIT else ""))

    # 分文件缓冲(含已完成)
    buf = {}
    for (f, cond, a), r in done.items():
        buf.setdefault(f, []).append(r)

    def save():
        with lock:
            for f, rows in buf.items():
                json.dump([{k:v for k,v in r.items() if k != '_kind'} for r in rows],
                          open(f,'w'), ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(max_workers=PAR) as ex:
        futs = {ex.submit(call, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                rec = fut.result()
                buf.setdefault(outfile(rec['_kind'], rec['cond']), []).append(rec)
                o = rec.get('out')
                shown = o if o is not None else ('⚠️' + repr((rec.get('raw') or '')[:30]))
                print(f"[{i}/{len(todo)}] {rec['_kind']:6} {rec['cond']:12} {rec['a']} 锚{rec.get('锚','-')} → {shown}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {t[0]} {t[1]} {t[2]['CLI']} → ERR {repr(e)}", flush=True)
            if i % 30 == 0: save()
    save()
    # 汇总
    print(f"\n完成。落盘:")
    for f in sorted(buf):
        rows = buf[f]; ok = sum(1 for r in rows if r.get('out') is not None)
        print(f"  {os.path.basename(f):26} 共{len(rows)} 有效out={ok}")
