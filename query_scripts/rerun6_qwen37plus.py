#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rerun6_qwen37plus.py — 补跑 rerun7 漏掉的模型:用重洗后修正的一审额当锚,重跑 6 个一审额抽错的正金额案(真锚×1 + 4 扰动 = 30 次)。
Qwen3.7-plus 非思考(DashScope;国际站用 RUN_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1);除 run_all.py 默认的 thinking:disabled 外另加 enable_thinking:False,确保非思考。第 7 个变动案 575835292 一审为 0,原锚未错,不跑。
输出 ../results/rerun6_qwen37plus.json。回车贴 key(不显示、不留存)。"""
import os, re, json, sys, threading, getpass, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
CASES={c['CLI']:c for c in json.load(open(os.path.join(ROOT,'案件_165案_语义修复版.json')))['cases']}
NEW={r['CLI']:r for r in json.load(open(os.path.join(ROOT,'复核','重洗结果.json')))}
P=json.load(open(os.path.join(ROOT,'prompts.json')))
SIX=['CLI.C.565931868','CLI.C.573164263','CLI.C.573703201','CLI.C.573956345','CLI.C.574048559','CLI.C.574750800']
MODEL='qwen3.7-plus'; BASE=os.environ.get('RUN_BASE','https://dashscope.aliyuncs.com/compatible-mode/v1'); TAG='qwen37plus'
EXTRA={'thinking':{'type':'disabled'},'enable_thinking':False}
OUT=os.path.join(ROOT,'results',f'rerun6_{TAG}.json')
CONDS={'真锚x1':1.0,'真扰动x1/3':1/3.0,'真扰动x2/3':2/3.0,'真扰动x3/2':1.5,'真扰动x3':3.0}
AP={'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉','原审原告、原审被告上诉':'双方上诉'}
def txt(f):
    if isinstance(f,dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')
def build(cond,cli):
    c=CASES[cli]; b=float(NEW[cli]['一审金额'])
    a=round(b*CONDS[cond],2); a=int(a) if a==int(a) else a
    M=P['模板']['带锚二审']
    诉请=txt(c['一审原告诉请']); 查明=txt(c['一审认定事实']); 新增=txt(c['二审新增认定事实'])
    ap=AP.get(c.get('二审上诉方类型'),c.get('二审上诉方类型',''))
    return M['指令'], M['输入_带锚'].format(诉请=诉请,一审查明=查明,二审新增=新增,锚=a,上诉方=ap), a
def parse_num(t):
    t=(t or '').strip().replace(',','').replace('，','')
    m=re.search(r'-?\d+\.?\d*',t)
    if m: return float(m.group())
    if re.search(r'true|驳回|不予支持|不承担|无需赔偿|应当?是?0',t,re.I): return 0.0
    return None
from openai import OpenAI
client=None
def call(cond,cli):
    sysmsg,usr,a=build(cond,cli)
    r=client.chat.completions.create(model=MODEL,temperature=0,
        messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}],extra_body=EXTRA)
    ch=r.choices[0] if getattr(r,'choices',None) else None
    raw=getattr(getattr(ch,'message',None),'content',None)
    if raw is None: raw=f"(空 finish={getattr(ch,'finish_reason','?')})"
    return {"a":cli,"cond":cond,"锚":a,"out":parse_num(raw),"raw":raw,"model":getattr(r,'model',MODEL),
            "run_at":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
if __name__=="__main__":
    key=getpass.getpass(f"贴 {TAG} 的 API key(不显示、不留存): ").strip()
    if not key: sys.exit("无 key,退出")
    client=OpenAI(api_key=key,base_url=BASE,timeout=180,max_retries=6)
    tasks=[(cond,cli) for cli in SIX for cond in CONDS]
    print(f"{MODEL} @ {BASE} 补跑 6 案 × 5 条件 = {len(tasks)} 次 ...")
    res=[]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(call,cond,cli):(cond,cli) for cond,cli in tasks}
        for f in as_completed(futs):
            cond,cli=futs[f]
            try:
                rec=f.result(); res.append(rec)
                print(f"  {cond} {cli} 锚{rec['锚']} -> {rec['out']}",flush=True)
            except Exception as e: print(f"  ERR {cond} {cli}: {e!r}",flush=True)
    json.dump(res,open(OUT,'w'),ensure_ascii=False,indent=1)
    print(f"完成 {len(res)}/{len(tasks)} 条 -> {OUT}")
