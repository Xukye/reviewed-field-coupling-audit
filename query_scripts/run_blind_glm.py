#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_blind_mimo.py — 主试验:盲一审 / 盲二审,模型只输出一个数字。
智谱GLM API(open.bigmodel.cn),默认 mimo-v2.5-pro 非思考、T=0、增量可续跑。

官方要点(已核 mimo.mi.com 文档):
  base_url       = https://api.xiaomimimo.com/v1
  鉴权           = Authorization: Bearer <key>(OpenAI SDK 默认)
  model          = mimo-v2.5-pro(混合推理,默认开思考)
  关思考         = enable_thinking:false(经 extra_body);官方 OpenAI 兼容页另有
                   thinking:{type:disabled} 写法 —— 用 THINK_OFF 切换。
  关思考后 T 可设(思考档 T 锁 1.0,故必须关思考才能 T=0)。

用法:
  export RUN_KEY=你的MiMo官方key
  # 先冒烟测 3 个,看 raw 是否纯数字、是否真关了思考:
  RUN_LIMIT=3 python3 run_blind_mimo.py 盲一审 盲二审
  # 没问题再全量 165:
  python3 run_blind_mimo.py 盲一审 盲二审

  # 若 enable_thinking:false 不生效(raw 里仍有思考过程),换写法:
  THINK_OFF=thinking python3 run_blind_mimo.py 盲一审 盲二审

结果存 ../results/blind1_<TAG>.json、blind2_<TAG>.json
每条:{a:溯源码, cond, out:数字, raw:原文, model:response.model}
"""
import os, re, json, sys, threading, getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE  = os.path.dirname(os.path.abspath(__file__))     # .../主试验/scripts
ROOT  = os.path.dirname(HERE)                          # .../主试验
RESDIR= os.path.join(ROOT, "results"); os.makedirs(RESDIR, exist_ok=True)
CASES = json.load(open(os.path.join(ROOT, '案件_165案_语义修复版.json')))['cases']
P     = json.load(open(os.path.join(ROOT, 'prompts.json')))

MODEL = os.environ.get("RUN_MODEL", "glm-5.1")
BASE  = os.environ.get("RUN_BASE",  "https://open.bigmodel.cn/api/paas/v4")
KEY   = os.environ.get("RUN_KEY", "").strip()
TAG   = os.environ.get("RUN_TAG", "glm51").strip()
PAR   = max(1, int(os.environ.get("RUN_PARALLEL", "30")))
LIMIT = int(os.environ.get("RUN_LIMIT", "0"))          # >0:只跑前 N(冒烟)
# 关思考写法(官方文档=thinking:{type:disabled}):thinking(默认,官方)/ enable / both
TOFF  = os.environ.get("THINK_OFF", "thinking").lower()
if   TOFF == "thinking": EXTRA = {"thinking": {"type": "disabled"}}
elif TOFF == "both":     EXTRA = {"enable_thinking": False, "thinking": {"type": "disabled"}}
else:                    EXTRA = {"enable_thinking": False}

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
    r = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}],
        extra_body=EXTRA)
    raw = r.choices[0].message.content
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
        KEY = getpass.getpass("请输入 智谱GLM API key(输入时不显示、运行结束不留存): ").strip()
    if not KEY:
        sys.exit("未输入 key,退出。")
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=180, max_retries=6)
    print(f"模型={MODEL} 端点={BASE} 关思考={TOFF}({EXTRA}) T=0 并发={PAR}"
          + (f" 小样本前{LIMIT}个" if LIMIT else " 全量"))
    for cond in conds:
        run_cond(cond)
