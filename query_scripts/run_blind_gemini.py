#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_blind_gemini.py — 主试验:盲一审 / 盲二审,模型只输出一行 'T 0' 或 'F 数字'。
Google Gemini 的 OpenAI 兼容端点,默认 gemini-2.5-flash 关思考、T=0、增量可续跑。

官方要点(Google AI / Gemini OpenAI 兼容):
  base_url   = https://generativelanguage.googleapis.com/v1beta/openai/
  鉴权       = Authorization: Bearer <AI Studio key>(OpenAI SDK 默认)
  model      = gemini-2.5-flash(默认开思考)
  关思考     = thinking_budget=0(经 extra_body.google.thinking_config);
               2.5 Flash 可设 0 关闭;另有 reasoning_effort="none" 写法 —— THINK_OFF 切换。
  关思考后 T 才确定(开思考会引入采样)。temperature=0。

用法:
  # 先冒烟测 3 个,看 raw 是否纯 'F 数字' / 'T 0'、是否真关了思考(无思考过程外泄):
  RUN_LIMIT=3 python3 "/Users/xukye/Library/Mobile Documents/iCloud~md~obsidian/Documents/TABLE/LAW/Idea/AI一审二审锚定效应/主试验/scripts/run_blind_gemini.py" 盲一审 盲二审
  # 没问题再全量 165×2:
  python3 "/Users/xukye/Library/Mobile Documents/iCloud~md~obsidian/Documents/TABLE/LAW/Idea/AI一审二审锚定效应/主试验/scripts/run_blind_gemini.py" 盲一审 盲二审

  # 若 thinking_budget 写法不被接受,换 reasoning_effort:
  THINK_OFF=effort python3 ".../run_blind_gemini.py" 盲一审 盲二审

结果存 ../results/blind1_<TAG>.json、blind2_<TAG>.json
每条:{a:溯源码, cond, out:数字, raw:原文, model:response.model}
回车后弹出隐藏输入贴 key(不留存,方便 .py 上传)。
"""
import os, re, json, sys, threading, getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE  = os.path.dirname(os.path.abspath(__file__))     # .../主试验/scripts
ROOT  = os.path.dirname(HERE)                          # .../主试验
RESDIR= os.path.join(ROOT, "results"); os.makedirs(RESDIR, exist_ok=True)
CASES = json.load(open(os.path.join(ROOT, '案件_165案_语义修复版.json')))['cases']
P     = json.load(open(os.path.join(ROOT, 'prompts.json')))

MODEL = os.environ.get("RUN_MODEL", "gemini-2.5-flash")
BASE  = os.environ.get("RUN_BASE",  "https://generativelanguage.googleapis.com/v1beta/openai/")
KEY   = os.environ.get("RUN_KEY", "").strip()
TAG   = os.environ.get("RUN_TAG", "gemini25flash").strip()
PAR   = max(1, int(os.environ.get("RUN_PARALLEL", "20")))
LIMIT = int(os.environ.get("RUN_LIMIT", "0"))          # >0:只跑前 N(冒烟)
# 关思考写法:budget(默认,thinking_budget=0)/ effort(reasoning_effort=none)/ both
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

FNAME = {"盲一审": "blind1", "盲二审": "blind2"}
AP = {'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉',
      '原审原告、原审被告上诉':'双方上诉'}

def txt(f):
    if isinstance(f, dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')

def build(cond, c):
    M = P['模板']
    诉请=txt(c['一审原告诉请']); 查明=txt(c['一审认定事实']); 新增=txt(c['二审新增认定事实'])
    ap=AP.get(c.get('二审上诉方类型'), c.get('二审上诉方类型',''))
    if cond == '盲一审':
        return M['盲一审']['指令'], M['盲一审']['输入'].format(诉请=诉请, 一审查明=查明)
    if cond == '盲二审':
        return M['盲二审']['指令'], M['盲二审']['输入'].format(诉请=诉请, 一审查明=查明, 二审新增=新增, 上诉方=ap)
    raise SystemExit(f"本脚本只支持 盲一审/盲二审,收到:{cond}")

def parse_num(t):
    """T/F 格式:'T 0'→0,'F 数字'→数字;纯数字兜底。返回 数字 或 None。"""
    s = (t or '').strip()
    head = s[:1].upper()
    m = re.search(r'-?\d+\.?\d*', s.replace(',', '').replace('，', ''))
    num = float(m.group()) if m else None
    if head == 'T': return 0.0
    if head == 'F': return num
    if num is not None: return num
    if re.search(r'驳回|不予支持|不承担|无需赔偿', s, re.I): return 0.0
    return None

try:
    from openai import OpenAI
except ImportError:
    sys.exit("请先 pip install openai")
client = None   # 运行时输入 key 后初始化
lock = threading.Lock()

def call(cond, c):
    sysmsg, usr = build(cond, c)
    kwargs = dict(model=MODEL, temperature=0,
                  messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}])
    if EXTRA:   kwargs["extra_body"] = EXTRA
    if REFFORT: kwargs["reasoning_effort"] = REFFORT
    r = client.chat.completions.create(**kwargs)
    ch  = r.choices[0] if getattr(r, 'choices', None) else None
    raw = getattr(getattr(ch, 'message', None), 'content', None)
    if raw is None:   # 空响应:不崩,记 finish_reason,out=None → 下次续跑重试
        raw = f"(空响应 finish_reason={getattr(ch, 'finish_reason', '?')})"
    return {"a": c['CLI'], "cond": cond, "out": parse_num(raw),
            "raw": raw, "model": getattr(r, 'model', MODEL)}

def run_cond(cond):
    outpath = os.path.join(RESDIR, f"{FNAME[cond]}_{TAG}.json")
    done = {r['a']: r for r in json.load(open(outpath)) if r.get('out') is not None} if os.path.exists(outpath) else {}
    pool = CASES[:LIMIT] if LIMIT > 0 else CASES
    todo = [c for c in pool if c['CLI'] not in done]
    print(f"[{cond}] 已完成 {len(done)} / 待跑 {len(todo)}")
    res = list(done.values())
    def save():
        with lock:
            json.dump(res, open(outpath,'w'), ensure_ascii=False, indent=1)
    with ThreadPoolExecutor(max_workers=PAR) as ex:
        futs = {ex.submit(call, cond, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                rec = f.result(); res.append(rec)
                out = rec.get('out')
                if out is None:
                    snip = (rec.get('raw') or '')[:60].replace('\n', ' ')
                    print(f"[{cond} {i}/{len(todo)}] {rec['a']} → ⚠️解析失败 raw='{snip}'", flush=True)
                else:
                    print(f"[{cond} {i}/{len(todo)}] {rec['a']} → {out}", flush=True)
            except Exception as e:
                print(f"[{cond} {i}/{len(todo)}] {c['CLI']} → ERR {repr(e)}", flush=True)
            if i % 20 == 0:
                save()
    save()
    zn = sum(1 for r in res if r.get('out') == 0)
    pn = sum(1 for r in res if r.get('out') not in (0, None))
    print(f"[{cond}] 完成 {len(res)}:输出0 {zn} / 正数 {pn} → {outpath}")

if __name__ == "__main__":
    conds = sys.argv[1:] or ['盲一审','盲二审']
    if not KEY:
        KEY = getpass.getpass("请输入 Gemini(AI Studio) API key(输入时不显示、运行结束不留存): ").strip()
    if not KEY:
        sys.exit("未输入 key,退出。")
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=180, max_retries=6)
    print(f"模型={MODEL} 端点={BASE} 关思考={TOFF}(extra={EXTRA},reasoning_effort={REFFORT}) T=0 并发={PAR}"
          + (f" 小样本前{LIMIT}个" if LIMIT else " 全量"))
    for cond in conds:
        run_cond(cond)
