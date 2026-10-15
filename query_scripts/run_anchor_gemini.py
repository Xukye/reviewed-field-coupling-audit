#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_anchor_gemini.py — 真锚×1 + 真扰动×{1/3,2/3,3/2,3}(锚=真实一审额×倍数)。
Google Gemini OpenAI 兼容端点,gemini-2.5-flash 关思考、T=0,带锚直问"判多少"。
★ 正锚150×5 + 零锚15(真锚x1,锚=0)= 765 全部混成一个大池子并发,落一个大 JSON。增量可续。

关思考 = thinking_budget=0(extra_body.google.thinking_config);THINK_OFF 可切 effort/both。
temperature=0。

用法:
  python3 ".../run_anchor_gemini.py"                 # 全部混跑(765)
  RUN_LIMIT=3 python3 ".../run_anchor_gemini.py"     # 每条件前 3 个(冒烟)
回车后弹出隐藏输入贴 key(不留存)。
结果存 ../results/anchor_all_gemini25flash.json,每条:{a,cond,锚,out,raw,model}
"""
import os, re, json, sys, threading, getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE  = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RESDIR= os.path.join(ROOT, "results"); os.makedirs(RESDIR, exist_ok=True)
CASES = json.load(open(os.path.join(ROOT, '案件_165案_语义修复版.json')))['cases']
P     = json.load(open(os.path.join(ROOT, 'prompts.json')))

MODEL = os.environ.get("RUN_MODEL", "gemini-2.5-flash")
BASE  = os.environ.get("RUN_BASE",  "https://generativelanguage.googleapis.com/v1beta/openai/")
KEY   = os.environ.get("RUN_KEY", "").strip()
TAG   = os.environ.get("RUN_TAG", "gemini25flash").strip()
PAR   = max(1, int(os.environ.get("RUN_PARALLEL", "20")))
LIMIT = int(os.environ.get("RUN_LIMIT", "0"))
TOFF  = os.environ.get("THINK_OFF", "budget").lower()
# 安全等级全设 BLOCK_NONE:法律案件含伤情/责任描述,易被安全过滤误杀致空响应
_SAFE = [{"category": c, "threshold": "BLOCK_NONE"} for c in
         ("HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH",
          "HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT")]
_BUDGET = {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0, "include_thoughts": False},
                                     "safety_settings": _SAFE}}}
if   TOFF == "budget": EXTRA, REFFORT = _BUDGET, None
elif TOFF == "effort": EXTRA, REFFORT = {}, "none"
else:                  EXTRA, REFFORT = _BUDGET, "none"
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
    kwargs = dict(model=MODEL, temperature=0,
                  messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}])
    if EXTRA:   kwargs["extra_body"] = EXTRA
    if REFFORT: kwargs["reasoning_effort"] = REFFORT
    r = client.chat.completions.create(**kwargs)
    ch  = r.choices[0] if getattr(r, 'choices', None) else None
    raw = getattr(getattr(ch, 'message', None), 'content', None)
    if raw is None:   # 空响应:不崩,记 finish_reason,out=None → 下次续跑重试
        raw = f"(空响应 finish_reason={getattr(ch, 'finish_reason', '?')})"
    return {"a": c['CLI'], "cond": cond, "锚": a, "out": parse_num(raw),
            "raw": raw, "model": getattr(r, 'model', MODEL)}

if __name__ == "__main__":
    conds = sys.argv[1:] or list(CONDS.keys())
    if not KEY:
        KEY = getpass.getpass("请输入 Gemini(AI Studio) API key(输入时不显示、运行结束不留存): ").strip()
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
    print(f"模型={MODEL} 关思考={TOFF} T=0 并发={PAR} | 正锚{len(pos)}×{len(conds)} + 零锚{len(zero)} = 大盘 {len(tasks_all)} | 已完成 {len(done)} / 待跑 {len(tasks)} → {OUT}")

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
    print(f"\n完成 {len(res)} 条 → {OUT}")
    for cond in conds:
        sub = [r for r in res if r['cond'] == cond]
        z = sum(1 for r in sub if r.get('out') == 0); pos = sum(1 for r in sub if r.get('out') not in (0, None)); bad = sum(1 for r in sub if r.get('out') is None)
        print(f"  {cond}: {len(sub)} | 0值 {z} / 正数 {pos} / 失败 {bad}")
