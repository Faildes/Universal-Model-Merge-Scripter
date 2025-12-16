import os, random, json, shlex
from string import Template
from typing import List

hexchars = "0123456789abcdef"
rnm = lambda n: ''.join(random.choices(hexchars, k=n))
_uid = lambda: f"{rnm(8)}-{rnm(4)}-{rnm(4)}-{rnm(4)}-{rnm(12)}"
def data_construct(lines):
    return [json.dumps({"cell_type":"code","execution_count":None,"id":_uid(),
                        "metadata":{"editable":True,"slideshow":{"slide_type":""},"tags":[]},
                        "outputs":[],"source":[x+"\n" for x in s.splitlines()]}) for s in lines]

def _ensure_dirs(root: str, subdirs: List[str]):
    for d in [root, *[os.path.join(root, x) for x in subdirs]]:
        os.makedirs(d, exist_ok=True)
        
def _nb_json(cells: List[str]) -> str:
    def wrap(src: str) -> str:
        return json.dumps({
            "cell_type":"code","execution_count":None,"id":"",
            "metadata":{},"outputs":[],
            "source":[line+"\n" for line in src.splitlines()]
        })
    return '{{"cells":[{cells}],"metadata":{{"kernelspec":{{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}},"language_info":{{"name":"python","version":"3.10.6"}}}},"nbformat":4,"nbformat_minor":5}}'.format(
        cells=",".join(map(wrap,cells))
    )

def _split(s: str):
    return shlex.split(s, posix=True)

def _needs_quote(val: str) -> bool:
    return ("," in val) or (" " in val)

def _ab_opt(flag: str, val: str, is_rand: bool) -> str:
    # flag: 'alpha' or 'beta'
    name = f"rand_{flag}" if is_rand else flag
    v = f'"{val}"' if _needs_quote(val) else val
    return f"--{name} {v}"

def _parse_tail_at(tokens):
    out = {"cosine": None, "fine": None, "seed": None, "mode": None, "extras": []}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("@"):
            out["extras"].append(t); i += 1; continue
        k, v = t[1:].lower(), None
        if "=" in k:
            k, v = k.split("=", 1)
            v = v.strip('"').strip("'")
        else:
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("@"):
                v = tokens[i + 1].strip('"').strip("'"); i += 1

        if k in ("cosine0", "cosine1", "cosine2"):
            out["cosine"] = int(k[-1])
        elif k in ("c","cosine") and v is not None:
            out["cosine"] = int(v)
        elif k in("f","fine") and v is not None:
            out["fine"] = v
        elif k in ("s", "seed") and v is not None:
            out["seed"] = int(v)
        elif k in ("m", "mode") and v is not None:
            v_norm = v.upper()
            out["mode"] = v_norm
        else:
            out["extras"].append(tokens[i])
        i += 1
    return out

def planit(filepath, workpath):
    res, final, last = [], None, None
    temp = lambda x: f"TEMP{x}" if x and x[0]=="_" else x
    mdl = lambda p: f'{workpath}/tmp/models/'
    vae = lambda : f'{workpath}/tmp/vae/VAE.safetensors'
    def emit(cmd,out_,has_next):
        nonlocal final
        final=out_; r=cmd+"\nflush()"+(f'\n\n{out_}=model("{out_}",1)' if has_next else "")
        res.append(r)
    def opts(cos,fine,seed,need_seed=False):
        s=[]
        if cos is not None: s.append(f"--cosine {cos}")
        if fine: s.append(f'--fine "{fine}"')
        if need_seed:
            sd = seed if seed is not None else random.randrange(2**63)
            s.append(f"--seed {sd}")
        return (" \\\n"+"\n".join(s)) if s else ""

    with open(filepath,"r+",encoding="utf-8") as f:
        line=f.readline()
        while line:
            t=line.rstrip("\n").replace("“",'"').replace("”",'"')
            nxt=f.readline(); has_next=bool(nxt)

            if not t.strip(): res.append(""); line=nxt; continue
            if t.startswith("//"): res.append("#"+t[2:]); line=nxt; continue

            # + (download/custom_model)
            if t.startswith("+"):
                last="download"
                if not (res and ("https://" in res[-1] or res[-1]=="")): res.append("")
                d_raw=t[1:]; d=[d_raw] if "," not in d_raw else d_raw.replace(" ","").split(",")
                mode="lora" if "%LR" in t else "checkpoint"
                if d[0].startswith("_"): d[0]=temp(d[0])
                if len(d)==1: res.append(f'{d[0]} = model("{d[0]}",1)')
                elif "/api/" in t:
                    res.append(f'{d[0]} = old_custom_model("{d[1]}","{d[0]}",1,"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR")'); final=d[0]
                else:
                    res.append(f'{d[0]} = custom_model("{d[1]}","{d[0]}",mode="{mode}")'); final=d[0]
                line=nxt; continue

            # CM (merge)
            if t.startswith("CM"):
                if last != "merge": res.append("flush()")
                last = "merge"; res.append("")

                toks = _split(t[2:].strip())
                if len(toks) < 3:
                    res.append("# error: CM needs at least: CM A op B ..."); line = nxt; continue

                cut = len(toks)
                for i, tk in enumerate(toks):
                    if tk.startswith("@") and tk.lower() not in ("@r", "@rand"):
                        cut = i; break
                core, at = toks[:cut], _parse_tail_at(toks[cut:])
                tail_opts = []
                if at["cosine"] is not None: tail_opts.append(f"--cosine{at['cosine']}")
                if at["fine"]: tail_opts.append(f'--fine {"\""+at["fine"]+"\"" if _needs_quote(at["fine"]) else at["fine"]}')
                if at["seed"] is not None: tail_opts.append(f"--seed {at['seed']}")
                tail_str = "" if not tail_opts else " \\\n" + "\n".join(tail_opts)
                at_mode = at["mode"]

                A = temp(core[0]); op1 = core[1].upper()

                def vae_path():  return f'{workpath}/tmp/vae/VAE.safetensors'
                def models_dir(): return f'{workpath}/tmp/models/'
                def emit(cmd, out_name, has_next):
                    nonlocal final
                    final = out_name
                    r = cmd + "\nflush()" + (f'\n\n{out_name}=model("{out_name}",1)' if has_next else "")
                    res.append(r)

                try:
                    if op1 == "+":  # WS / TRS / ST
                        if len(core) < 5:
                            res.append("# error: CM A + B alpha result ..."); line = nxt; continue
                        B = temp(core[2])
                        if core[3].upper() in ("+T","+S"):
                            op2 = core[3].upper()
                            if len(core) < 8:
                                res.append("# error: CM A + B +T|+S C alpha beta result"); line = nxt; continue
                            C = temp(core[4])

                            # α
                            is_ra = core[5].lower() in ("@r","@rand")
                            a_val = core[6] if is_ra else core[5]
                            # β
                            idx_b = 7 if is_ra else 6
                            is_rb = core[idx_b].lower() in ("@r","@rand")
                            b_val = core[idx_b+1] if is_rb else core[idx_b]
                            # out
                            out_  = temp(core[idx_b + (2 if is_rb else 1)])

                            kind = at_mode if at_mode else ("TRS" if op2 == "+T" else "ST")
                            opts = []
                            opts.append(_ab_opt("alpha", a_val, is_ra))
                            opts.append(_ab_opt("beta",  b_val, is_rb))
                            extra = at

                            cmd = (
                                f'!python merge.py "{kind}" "{models_dir()}" "{A}.safetensors" "{B}.safetensors" --model_2 "{C}.safetensors" \\\n'
                                f'--vae "{vae_path()}" \\\n'
                                + " \\\n".join(opts) + " \\\n"
                                f'--save_half --prune --save_safetensors --output "{out_}"' + tail_str
                            )
                            emit(cmd, out_, bool(nxt))

                        else:
                            # WS: A + B alpha result
                            is_ra = core[3].lower() in ("@r","@rand")
                            a_val = core[4] if is_ra else core[3]
                            out_  = temp(core[5] if is_ra else core[4])
                            
                            kind = at_mode if at_mode else "WS"

                            cmd = (
                                f'!python merge.py "{kind}" "{models_dir()}" "{A}.safetensors" "{B}.safetensors" \\\n'
                                f'--vae "{vae_path()}" \\\n'
                                f'{_ab_opt("alpha", a_val, is_ra)} \\\n'
                                f'--save_half --prune --save_safetensors --output "{out_}"' + tail_str
                            )
                            emit(cmd, out_, bool(nxt))

                    elif op1 == "+D":  # DARE
                        if len(core) < 6:
                            res.append("# error: CM A +D B alpha beta result"); line = nxt; continue
                        B = temp(core[2])
                        # α
                        is_ra = core[3].lower() in ("@r","@rand")
                        a_val = core[4] if is_ra else core[3]
                        # β
                        idx_b = 5 if is_ra else 4
                        is_rb = core[idx_b].lower() in ("@r","@rand")
                        b_val = core[idx_b+1] if is_rb else core[idx_b]
                        # out
                        out_  = temp(core[idx_b + (2 if is_rb else 1)])
                        kind = at_mode if at_mode else "DARE"
                        cmd = (
                            f'!python merge.py "{kind}" "{models_dir()}" "{A}.safetensors" "{B}.safetensors" \\\n'
                            f'--vae "{vae_path()}" \\\n'
                            f'{_ab_opt("alpha", a_val, is_ra)} \\\n'
                            f'{_ab_opt("beta",  b_val, is_rb)} \\\n'
                            f'--save_half --prune --save_safetensors --output "{out_}"' + tail_str
                        )
                        emit(cmd, out_, bool(nxt))

                    elif op1 == "#S":  # SWAP
                        if len(core) < 5:
                            res.append("# error: CM A #S B alpha result"); line = nxt; continue
                        B = temp(core[2])
                        is_ra = core[3].lower() in ("@r","@rand")
                        a_val = core[4] if is_ra else core[3]
                        out_  = temp(core[5] if is_ra else core[4])
                        kind = at_mode if at_mode else "SWAP"
                        cmd = (
                            f'!python merge.py "{kind}" "{models_dir()}" "{A}.safetensors" "{B}.safetensors" \\\n'
                            f'--vae "{vae_path()}" \\\n'
                            f'{_ab_opt("alpha", a_val, is_ra)} \\\n'
                            f'--save_half --prune --save_safetensors --output "{out_}"' + tail_str
                        )
                        emit(cmd, out_, bool(nxt))

                    elif op1 == "#X":  # CLIPXOR
                        if len(core) < 4:
                            res.append("# error: CM A #X B result"); line = nxt; continue
                        B = temp(core[2]); out_ = temp(core[3])
                        kind = at_mode if at_mode else "CLIPXOR"
                        cmd = (
                            f'!python merge.py "{kind}" "{models_dir()}" "{A}.safetensors" "{B}.safetensors" \\\n'
                            f'--vae "{vae_path()}" \\\n'
                            f'--save_half --prune --save_safetensors --output "{out_}"' + tail_str
                        )
                        emit(cmd, out_, bool(nxt))
                        
                    elif op1 == "+F":  # FWM
                        if len(core) < 5:
                            res.append("# error: CM A +F B alpha result"); line = nxt; continue
                        B = temp(core[2])
                        
                        is_ra = core[3].lower() in ("@r","@rand")
                        a_val = core[4] if is_ra else core[3]
                        
                        out_  = temp(core[5] if is_ra else core[4])
                        kind = at_mode if at_mode else "FWM"
                        cmd = (
                            f'!python merge.py "{kind}" "{models_dir()}" "{A}.safetensors" "{B}.safetensors" \\\n'
                            f'--vae "{vae_path()}" \\\n'
                            f'{_ab_opt("alpha", a_val, is_ra)} \\\n'
                            f'--save_half --prune --save_safetensors --output "{out_}"' + tail_str
                        )
                        emit(cmd, out_, bool(nxt))

                    else:
                        res.append("# error: unknown CM operator (use +, +D, #S, #X, and optional +T/+S)")
                except Exception as e:
                    res.append(f"# error: {e!r}")

                line = nxt
                continue

            # LB (lora_bake)
            if t.startswith("LB"):
                if last!="merge": res.append("flush()")
                last="merge"
                base,pairs,out_ = t[3:].split(" ")
                base,out_ = temp(base), temp(out_)
                fe = ",".join(q.replace(":",".safetensors:") for q in pairs.split(","))
                cmd = (f'!python lora_bake.py "{workpath}/tmp/models/" "{base}.safetensors" \\\n'
                       f'"{fe}" \\\n--save_half --prune --save_safetensors --output "{out_}"')
                emit(cmd,out_,has_next); line=nxt; continue

            # PR (prune pass-through)
            if t.startswith("PR"):
                if last!="merge": res.append("flush()")
                last="merge"
                a,out_ = t[3:].split(" ")
                a,out_ = temp(a), temp(out_)
                cmd = (f'!python merge.py "NoIn" "{workpath}/tmp/models/" "{a}.safetensors" None \\\n'
                       f'--vae "{vae()}" \\\n--save_half --prune --save_safetensors --output "{out_}"')
                emit(cmd,out_,has_next); line=nxt; continue

            # remove
            if t.startswith("-"):
                last="download"
                if not (res and ("remove_model" in res[-1] or res[-1]=="")): res.append("")
                res.append(f"remove_model({temp(t.replace('-',''))})"); line=nxt; continue

            line=nxt
    return res, final

INSTALL_TPL = Template(r"""!pip install torch torchvision lora fake_useragent diffusers torchsde git+https://github.com/huggingface/diffusers git+https://github.com/Faildes/sd_embed_negpip.git
!pip install -U peft
!pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121
!apt-get -y install -qq aria2
%cd $workpath/working/
!git clone https://github.com/Faildes/merge-models -b notebook
""")

PRELUDE_TPL = Template(r"""
#@title Model Merge / Model Download
import os, re, gc, json, shutil, hashlib, requests, torch, filelock
from fake_useragent import UserAgent

HFToken = "$hf_token"
CVToken = "$cv_token"

workpath = "$workpath"
models_dir = f"{workpath}/tmp/models"
vae_dir    = f"{workpath}/tmp/vae"
emb_dir    = f"{workpath}/tmp/embeddings"
for p in (f"{workpath}/tmp", models_dir, vae_dir, emb_dir): os.makedirs(p, exist_ok=True)

pref = {
    "format": "SafeTensor",
    "size": "pruned",
    "fp": "fp16"}

def flush(light=True):
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    if not light:
        import subprocess
        subprocess.run(["pip", "cache", "purge"])

def remove_model(path):
    print(f"Delete {os.path.basename(path)}")
    os.remove(path)
    total, used, free = shutil.disk_usage("/")
    print(f"Remain Storage: {free / (2**30):.2f}GB/{total / (2**30):.2f}GB")
    
def make_pref(p,mode):
    pref_set = {
        "size": ["full","pruned"],
        "fp": ["fp16","bf16","fp8","fp32"],
        "format": ["PickleTensor","SafeTensor"]}
    def lsrt(lst,odr):
        return [lst[i] for i in odr]
    if mode == "lora":
        return [{"format":"SafeTensor"},{"format":"PickleTensor"}]
    elif mode == "checkpoint":
        n = [pref_set[v].index(p[v]) for v in pref_set.keys()]
        mx=[1,2,1]
        srt = {}
        srt["size"] = lsrt(pref_set["size"],[1,0]) if n[0]==1 else pref_set["size"]
        if n[1] == 0:
            srt["fp"] = pref_set["fp"]
        elif n[1] == 1:
            srt["fp"] = lsrt(pref_set["fp"],[1,0,2])
        elif n[1] == 2:
            srt["fp"] = lsrt(pref_set["fp"],[2,0,1])
        srt["format"] = lsrt(pref_set["format"],[1,0]) if n[2]==1 else pref_set["format"]
        r=[]
        for i in range(len(pref_set["format"])):
            for j in range(len(pref_set["fp"])):
                for k in range(len(pref_set["size"])):
                    r.append([k,j,i])
        res=[]
        for i in r:
            f = {
                "size":srt["size"][i[0]],
                "fp":srt["fp"][i[1]],
                "format":srt["format"][i[2]]}
            res.append(f)
        return res
    else:
        return "ERROR"
        
# get meta list and search the pref
def get_dl(url, version:str =None, mode:str ="checkpoint"):
    prefs = make_pref(pref,mode)
    if "civitai"in url:
        cid=re.sub(r"\D", "", re.search("models/[0-9]+",url).group())
        if "Version" in url and version is None:
            version = re.sub(r"\D", "", re.search("modelVersionId=[0-9]+",url).group())
        api=f"https://civitai.com/api/v1/models/{cid}"
        response=requests.get(api)
        if response.status_code == 200:
            d=response.json()
            model_name=d["name"]
            model_version=version if version is not None else d["modelVersions"][0]["name"]
            for k in d["modelVersions"]:
                if k["name"] == model_version or str(k["id"]) == model_version:
                    model=k
                    model_version=k["name"]
                    break
            meta_list = [a["metadata"] for a in model["files"]]
            for p in prefs:
                try:
                    i = meta_list.index(p)
                    file = model["files"][i]
                    break
                except:
                    continue
            dllink=file["downloadUrl"]
            sha256=file["hashes"]["SHA256"].lower()
            ext = file["metadata"]["format"]
            if ext == "SafeTensor":
                ex = 1
            else:
                ex = 0
            dlname=model_name+"-"+model_version
            q = {"url":dllink,
                 "name":dlname,
                 "format":ex,
                 "sha256":sha256}
            return q     
        else:
            return None
    elif "hugging" in url:
        url_set = url.replace("https://huggingface.co/","").split("/")
        base="https://huggingface.co/"
        api=base
        dllink=base
        dname=url_set[-1].rsplit(".",1)
        dlname=dname[0]
        if dname[1] == "safetensors":
            ex = 1
        else:
            ex = 0
        for i,s in enumerate(url_set):
            if i == 2:
                api+="raw/"
                dllink+="resolve/"
            else:
                api+=f"{s}/"
                dllink+=f"{s}/"

        res = requests.get(api)
        if res.status_code == 200:
            d=res.text
            sha256=re.search("sha256:[0-9a-f]+",d).group().replace("sha256:","")
            q = {"url":dllink,
                 "name":dlname,
                 "format":ex,
                 "sha256":sha256}
            return q
        else:
            return None
            
cache_filename = os.path.join(models_dir, "cache.json")
cache_data = None

def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(f"{cache_filename}.lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s

def dump_cache():
    with filelock.FileLock(f"{cache_filename}.lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)

def sha256(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value

    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value

def calculate_shorthash(filename):
    sha256 = sha256(filename, f"checkpoint/{os.path.splitext(os.path.basename(filename))[0]}")
    if sha256 is None:
        return

    shorthash = sha256[0:10]

    return shorthash

def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)

    if title not in hashes:
        return None

    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)

    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None

    return cached_sha256

def addnet_hash_safetensors(b):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def sha256_set(filename, title, sha256_value, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    print(f"{filename}: {sha256_value}")

    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }

    dump_cache()


user_header = f"\"Authorization: Bearer {HFToken}\""

def model(name,format=0):
    ext = "ckpt" if format == 0 else "safetensors"
    sha256_set(f"{models_dir}/{name}.{ext}", f"checkpoint/{name}", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR")
    return f"{models_dir}/{name}.{ext}"

def custom_model(url, checkpoint_name=None, mode="checkpoint"):
  user_token = HFToken if "huggingface" in url else CVToken
  parse = {"url":url,"version":None, "mode":mode} if type(url) is not list else {"url":url[0],"version":url[1], "mode":mode}
  g = get_dl(**parse)
  url = g["url"]
  checkpoint_name = g["name"] if checkpoint_name is None else checkpoint_name
  sha256 = g["sha256"]
  format = g["format"]
  if format == 0:
    ext = "ckpt"
  elif format == 1:
    ext = "safetensors"
  if os.path.exists(f"{models_dir}/{checkpoint_name}.{ext}"):
    return f"{models_dir}/{checkpoint_name}.{ext}"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d {models_dir} -o {checkpoint_name}.{ext}
  else:
    headers = {
          'User-Agent': UserAgent().chrome,
          'Sec-Ch-Ua': '"Brave";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
          'Sec-Ch-Ua-Mobile': '?0',
          'Sec-Ch-Ua-Platform': '"Windows"',
          'Sec-Fetch-Dest': 'document',
          'Sec-Fetch-Mode': 'navigate',
          'Sec-Fetch-Site': 'none',
          'Sec-Fetch-User': '?1',
          'Sec-Gpc': '1',
          'Upgrade-Insecure-Requests': '1',
          'Authorization': f'Bearer {user_token}'
    }
    response = requests.get(url, headers=headers, allow_redirects=False)
    download_link = response.headers["Location"]
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{models_dir}" -o {checkpoint_name}.{ext}
  if sha256 is not None:
    sha256_set(f"{models_dir}/{checkpoint_name}.{ext}", f"{mode}/{checkpoint_name}", sha256)
  return f"{models_dir}/{checkpoint_name}.{ext}"

def old_custom_model(url, checkpoint_name=None, format=0, sha256=None):
  user_token = HFToken if "huggingface" in url else CVToken
  if format == 0:
    ext = "ckpt"
  elif format == 1:
    ext = "safetensors"
  if os.path.exists(f"{models_dir}/{checkpoint_name}.{ext}"):
    return f"{models_dir}/{checkpoint_name}.{ext}"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d {models_dir} -o {checkpoint_name}.{ext}
  else:
    headers = {
          'User-Agent': UserAgent().chrome,
          'Sec-Ch-Ua': '"Brave";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
          'Sec-Ch-Ua-Mobile': '?0',
          'Sec-Ch-Ua-Platform': '"Windows"',
          'Sec-Fetch-Dest': 'document',
          'Sec-Fetch-Mode': 'navigate',
          'Sec-Fetch-Site': 'none',
          'Sec-Fetch-User': '?1',
          'Sec-Gpc': '1',
          'Upgrade-Insecure-Requests': '1',
          'Authorization': f'Bearer {user_token}'
    }
    response = requests.get(url, headers=headers, allow_redirects=False)
    download_link = response.headers["Location"]
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{models_dir}" -o {checkpoint_name}.{ext}
  if sha256 is not None:
    sha256_set(f"{models_dir}/{checkpoint_name}.{ext}", f"checkpoint/{checkpoint_name}", sha256)
  return f"{models_dir}/{checkpoint_name}.{ext}"

def custom_vae(url, vae_name=None):
    user_token = HFToken if "huggingface" in url else CVToken
    if "civitai" in url:
        if "api" in url:
            ext = "safetensors" if "SafeTensor" in url else "ckpt"
            headers = {
                  'User-Agent': UserAgent().chrome,
                  'Sec-Ch-Ua': '"Brave";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
                  'Sec-Ch-Ua-Mobile': '?0',
                  'Sec-Ch-Ua-Platform': '"Windows"',
                  'Sec-Fetch-Dest': 'document',
                  'Sec-Fetch-Mode': 'navigate',
                  'Sec-Fetch-Site': 'none',
                  'Sec-Fetch-User': '?1',
                  'Sec-Gpc': '1',
                  'Upgrade-Insecure-Requests': '1',
                  'Authorization': f'Bearer {user_token}'
            }
            response = requests.get(url, headers=headers, allow_redirects=False)
            download_link = response.headers["Location"]
            !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{vae_dir}" -o {vae_name}.{ext}
        else:
            pref = ["SafeTensor", "PickleTensor"]
            cid=re.sub(r"\D", "", re.search("models/[0-9]+",url).group())
            if "Version" in url:
                version = re.sub(r"\D", "", re.search("modelVersionId=[0-9]+",url).group())
            else:
                version = None
            api=f"https://civitai.com/api/v1/models/{cid}"
            response=requests.get(api)
            if response.status_code == 200:
                d=response.json()
                model_name=d["name"] if vae_name is None else vae_name
                model_version=version if version is not None else d["modelVersions"][0]["name"]
                for k in d["modelVersions"]:
                    if k["name"] == model_version or str(k["id"]) == model_version:
                        model=k
                        model_version=k["name"]
                        break
                meta_list = [a["metadata"]["format"] for a in model["files"]]
                for p in pref:
                    try:
                        i = meta_list.index(p)
                        file = model["files"][i]
                        break
                    except:
                        continue
                dllink=file["downloadUrl"]
                ext = file["metadata"]["format"]
                if ext == "SafeTensor":
                    ex = "safetensors"
                else:
                    ex = "ckpt"
                headers = {
                      'User-Agent': UserAgent().chrome,
                      'Sec-Ch-Ua': '"Brave";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
                      'Sec-Ch-Ua-Mobile': '?0',
                      'Sec-Ch-Ua-Platform': '"Windows"',
                      'Sec-Fetch-Dest': 'document',
                      'Sec-Fetch-Mode': 'navigate',
                      'Sec-Fetch-Site': 'none',
                      'Sec-Fetch-User': '?1',
                      'Sec-Gpc': '1',
                      'Upgrade-Insecure-Requests': '1',
                      'Authorization': f'Bearer {user_token}'
                }
                response = requests.get(dllink, headers=headers, allow_redirects=False)
                download_link = response.headers["Location"]
                !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{vae_dir}" -o {model_name}.{ex}
                vae_name = model_name
                ext = ex
            else:
                print("ERROR: VAE Not Found")
                return None
    elif "huggingface" in url:
        user_header = f"\"Authorization: Bearer {user_token}\""
        ext = "safetensors" if "safetensors" in url else "ckpt"
        if "blob/main" in url:
            url = url.replace("blob/main","resolve/main")
        !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d {vae_dir} -o {vae_name}.{ext}
    return f"{vae_dir}/{vae_name}.{ext}"

custom_vae("$vae_url","VAE")

flush(light=False)
%cd $workpath/working/merge-models
""")

UPLOAD_TPL = Template(r"""
#@title Upload the model to huggingface
from huggingface_hub import upload_file
User_Repository = "$repo"
%cd $workpath/tmp/models
upload_file(path_or_fileobj="$workpath/tmp/models/$final.safetensors",
            path_in_repo="$final.safetensors",
            repo_id=User_Repository, token=HFToken)
!pip cache purge
""")

T2I_CFG_TPL = Template(r"""
#@title Pipe Config (short)
import os, gc, torch, diffusers
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

checkpoint="$final"
ext="safetensors"
model_type="fp16"
scheduler="euler_a"
vpred=False

SCHEDULERS = {
  "unipc":[diffusers.schedulers.UniPCMultistepScheduler,{{}},"UniPC"],
  "euler_a":[diffusers.schedulers.EulerAncestralDiscreteScheduler,{{}},"Euler a"],
  "euler":[diffusers.schedulers.EulerDiscreteScheduler,{{}},"Euler"],
  "ddim":[diffusers.schedulers.DDIMScheduler,{{}},"DDIM"],
  "ddpm":[diffusers.schedulers.DDPMScheduler,{{}},"DDPM"],
  "deis":[diffusers.schedulers.DEISMultistepScheduler,{{}},"DEIS"],
  "dpm2":[diffusers.schedulers.KDPM2DiscreteScheduler,{{}},"DPM2"],
  "dpm2_karras":[diffusers.schedulers.KDPM2DiscreteScheduler,{{"use_karras_sigmas":True}},"DPM2 Karras"],
  "dpm2-a":[diffusers.schedulers.KDPM2AncestralDiscreteScheduler,{{}},"DPM2 a"],
  "dpm2-a_karras":[diffusers.schedulers.KDPM2AncestralDiscreteScheduler,{{"use_karras_sigmas":True}},"DPM2 a Karras"],
  "dpm++_2s_a":[diffusers.schedulers.DPMSolverSinglestepScheduler,{{}},"DPM++ 2S a"],
  "dpm++_2s_a_karras":[diffusers.schedulers.DPMSolverSinglestepScheduler,{{"use_karras_sigmas":True}},"DPM++ 2S a Karras"],
  "dpm++_2m":[diffusers.schedulers.DPMSolverMultistepScheduler,{{}},"DPM++ 2M"],
  "dpm++_2m_karras":[diffusers.schedulers.DPMSolverMultistepScheduler,{{"use_karras_sigmas":True}},"DPM++ 2M Karras"],
  "dpm++_2m_sde":[diffusers.schedulers.DPMSolverMultistepScheduler,{{"algorithm_type":"sde-dpmsolver++"}},"DPM++ 2M SDE"],
  "dpm++_2m_sde_karras":[diffusers.schedulers.DPMSolverMultistepScheduler,{{"algorithm_type":"sde-dpmsolver++","use_karras_sigmas":True}},"DPM++ 2M SDE Karras"],
  "dpm++_sde":[diffusers.schedulers.DPMSolverSDEScheduler,{{}},"DPM++ SDE"],
  "dpm++_sde_karras":[diffusers.schedulers.DPMSolverSDEScheduler,{{"use_karras_sigmas":True}},"DPM++ SDE Karras"],
  "heun":[diffusers.schedulers.HeunDiscreteScheduler,{{}},"Heun"],
  "heun_karras":[diffusers.schedulers.HeunDiscreteScheduler,{{"use_karras_sigmas":True}},"Heun Karras"],
  "lms":[diffusers.schedulers.LMSDiscreteScheduler,{{}},"LMS"],
  "lms_karras":[diffusers.schedulers.LMSDiscreteScheduler,{{"use_karras_sigmas":True}},"LMS Karras"],
  "pndm":[diffusers.schedulers.PNDMScheduler,{{}},"PNDM"],
}
mt={"fp16":torch.float16,"fp32":torch.float32,"bf16":torch.bfloat16}

def flush(light=True):
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    if not light:
        import subprocess
        subprocess.run(["pip", "cache", "purge"])
    
cpath=f"$workpath/tmp/models/{checkpoint}.{ext}"
dtype=mt[model_type]
sch=SCHEDULERS[scheduler][1]

pipet=StableDiffusionXLPipeline.from_single_file(cpath, torch_dtype=dtype, use_safetensors=True, variant="fp16")
scd=SCHEDULERS[scheduler][0].from_config(pipet.scheduler.config, **sch)
scd_name=SCHEDULERS[scheduler][2]

pipet=StableDiffusionXLPipeline.from_single_file(cpath, torch_dtype=dtype, scheduler=scd, use_safetensors=True, variant="fp16")
pipet.safety_checker=None
pipe=pipet.to("cuda:0")
del pipet
flush(light=False)

refinert=StableDiffusionXLImg2ImgPipeline.from_single_file(cpath, torch_dtype=dtype, scheduler=scd, use_safetensors=True, variant="fp16")
refinert.safety_checker=None
refiner=refinert.to("cuda:1")
del refinert
flush()

init_pipe, init_refiner = pipe, refiner
""")

T2I_RUN_TPL = Template(r"""
#@title t2i
import os, gc, random, numpy, torch
from PIL import Image
from IPython.display import display
from PIL.PngImagePlugin import PngInfo
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
pipe, refiner = init_pipe, init_refiner

def color_balance(img: Image.Image, adj: dict) -> Image.Image:
    for k in ('shadow', 'middle', 'highlight'):
        v = adj.get(k)
        if not isinstance(v, (list, tuple)) or len(v) != 5:
            raise ValueError(f"adjustments['{k}'] must be a length-5 list or tuple")
    mode = img.mode
    if mode == 'RGBA':
        rgb, alpha = img.convert('RGB'), img.split()[-1]
    else:
        alpha = None
        rgb = img.convert('RGB') if mode != 'RGB' else img
    orig = numpy.array(rgb, dtype=numpy.float32)
    lum = orig.mean(axis=2)
    ws = numpy.clip((128.0 - lum) / 128.0, 0.0, 1.0)
    wh = numpy.clip((lum - 128.0) / 128.0, 0.0, 1.0)
    wm = 1.0 - ws - wh
    res = numpy.zeros_like(orig)
    for w, region in zip((ws, wm, wh), ('shadow', 'middle', 'highlight')):
        r, g, b, bright, contrast = adj[region]
        af = numpy.array([r, g, b], dtype=numpy.float32) / 100.0
        delta = (255.0 - orig) * numpy.maximum(af, 0.0) + orig * numpy.minimum(af, 0.0)
        rv = (orig + delta) * bright
        mean = rv.mean(axis=(0,1), keepdims=True)
        rv = (rv - mean) * contrast + mean
        rv = numpy.clip(rv, 0, 255)
        res += rv * w[..., None]
    out = Image.fromarray(numpy.clip(res, 0.0, 255.0).astype(numpy.uint8), 'RGB')
    if alpha is not None:
        out = Image.merge('RGBA', (*out.split(), alpha))
    del orig, res, rv, delta, lum, ws, wh, wm
    return out
    
def load_lora_weights(pipe, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
    # We could have accessed the unet config from `lora_state_dict()` too. We pass
    # it here explicitly to be able to tell that it's coming from an SDXL
    # pipeline.
    state_dict, network_alphas = pipe.lora_state_dict(
        pretrained_model_name_or_path_or_dict,
        unet_config=pipe.unet.config,
        **kwargs,
    )
    pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet)

    text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
    if len(text_encoder_state_dict) > 0:
        pipe.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=pipe.text_encoder,
            prefix="text_encoder",
            lora_scale=pipe.lora_scale,
        )

    text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
    if len(text_encoder_2_state_dict) > 0:
        pipe.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=pipe.text_encoder_2,
            prefix="text_encoder_2",
            lora_scale=pipe.lora_scale,
        )
        
def lora_prompt(prompt, pipe, refine, lhash):
    loras = []
    adap_list=[]
    alphas=[]
    add = []
    def network_replacement(m):
        alias = m.group(1)
        num = m.group(2)
        try:
            data = lpath[alias]
            mpath = data[1]
            dpath = data[0]
            add.append(data[2])
        except:
            return ""
        if "|" in num:
            t = num.split("|")
            alpha = float(t[0])
            apply = t[1]
            npath = f"{mpath}{alias}_{apply}.safetensors"
            try:
              data = lpath[f"{alias}_{apply}"]
              loras.append([data[0], alpha])
              return ""
            except:
              lpath[f"{alias}_{apply}"] = [npath, dpath]
              %cd /content/apply-lora-block-weight/
              !python apply_lora_block_weight.py {dpath} {npath} {apply}
              %cd /content/
              dpath = npath
        else:
            alpha = float(num)
        loras.append([dpath, alpha])
        return ""
    re_lora = re.compile("<lora:([^:]+):([^:]+)>")
    prompt = re.sub(re_lora, network_replacement, prompt)
    if loras == []:
        return prompt, lhash
    for k in add:
      if k not in prompt:
        prompt += ","+k
    for k in loras:
        p = os.path.abspath(os.path.join(k[0], ".."))
        safe = os.path.basename(k[0])
        name = os.path.splitext(safe)[0].replace(".","_")
        alphas.append(k[1])
        adap_list.append(name)
        try:
            pipe.load_lora_weights(p, weight_name=safe, adapter_name=name)
            refine.load_lora_weights(p, weight_name=safe, adapter_name=name)
        except:
            pass
        try:
            shash = lhash[name]
        except:
            lhash[name] = sha256(k[0], name, True)[0:10]
    pipe.set_adapters(adap_list, adapter_weights=alphas)
    refine.set_adapters(adap_list, adapter_weights=alphas)
    return prompt, lhash
    
def flush(light=True):
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    if not light:
        import subprocess
        subprocess.run(["pip", "cache", "purge"])

w,h = 768,1152
steps = 20
global_seed = -1
guidance=4.5
clip_skip=2
num_gen=4
adjust = {
    'shadow':    [0, 0, 0, 1.0, 1.0],    # Shadow RGBBrC
    'middle':    [0, 0, 0, 1.0, 1.0],     # Middle RGBBrC
    'highlight': [0, 0, 0, 1.0, 1.0],    # Highlight RGBBrC
}

hires=False
hires_scale=1.5
global_hires_seed = -2
hires_steps=40
guidance_h=4
denoise=0.4

prompt = "masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 1girl, blonde hair, long hair, floating hair, blue eyes, looking at viewer, parted lips, medium breasts, puffy sleeve white dress, leaning side against tree, dutch angle, upper body, (portrait, close-up:1.2), foreshortening, vines, green, forest, flowers, white butterfly, BREAK, dramatic shadow, depth of field, vignetting, dappled sunlight, lens flare, backlighting, volumetric lighting"
neg = "modern, recent, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, missing fingers, extra fingers, extra digits, fewer digits, cropped, very displeasing, (worst quality, bad quality:1.2), sketch, jpeg artifacts, signature, watermark, username, (censored, bar_censor, mosaic_censor:1.2), simple background, conjoined, bad ai-generated"

idir="$workpath/working/t2i_images/"
os.makedirs(idir, exist_ok=True)

def bpro(p):
    nl=[]; t=0; off=0
    for g in p.split(","):
        if "BREAK" in g: add=(t+off)%75; nl+=[" "]*add; off+=add; continue
        t+=g.count(" ")+1; nl.append(g)
    return ",".join(nl)

lhash = {}
pp, lhash = lora_prompt(prompt, pipe, refiner, lhash)
(emb, nemb, pool, npool) = get_weighted_text_embeddings_sdxl(pipe, prompt=bpro(pp), neg_prompt=neg)
global_seed = random.randrange(4294967294) if global_seed < 0 else global_seed
global_hires_seed = random.randrange(4294967294) if global_hires_seed < -1 else (global_seed if global_hires_seed < 0 else global_hires_seed)
print(global_seed)
if global_hires_seed != global_seed: print(global_hires_seed)
numpy.random.seed(global_seed)
seeds=numpy.random.randint(0,4294967294,num_gen)
numpy.random.seed(global_hires_seed)
hires_seeds=numpy.random.randint(0, 4294967294, num_gen)
if num_gen == 1:
    seeds = [global_seed]
    hires_seeds = [global_hires_seed]
disp=min(512/w, 512/h)
flat_adjust = f"(\"shadow\": {adjust['shadow']}, \"middle\": {adjust['middle']}, \"highlight\": {adjust['highlight']})"

for i,s in enumerate(seeds):
    if hires: hs = hires_seeds[i]
    else: hs = s
    gen = torch.Generator("cpu").manual_seed(int(s))
    genh = torch.Generator("cpu").manual_seed(hs)
    info=f"{prompt}\nNegative prompt: {neg}\nSteps: {steps}, Sampler: {scd_name}, CFG scale: {guidance}, Seed: {s}, Global Seed: {global_seed}, Size: {w}x{h}, Clip skip: {clip_skip}, Model: {checkpoint}"
    if hires:
        geninfo += f"{f', Hires Global Seed: {global_hires_seed}, Hires Seed: {hs}, ' if global_hires_seed != global_seed else ''}, Hires steps: {hires_steps}, Hires upscale: {hires_scale}, {f'Hires Adjust: {flat_adjust}, ' if any(c != [0]*3+[1.0]*2 for c in adjust.values()) else ''}Denoising strength: {denoise}, Hires CFG Scale: {guidance_h}"
    if len(lhash) > 0:
        geninfo += ", Lora hashes: \""
        n = ""
        for q, u in lhash.items():
        n += f"{q}: {u}, "
        n = n[:-2]
        geninfo += f"{n}\""
    meta=PngInfo()
    meta.add_text("parameters", info)
    img=pipe(prompt_embeds=emb, pooled_prompt_embeds=pool,
            negative_prompt_embeds=nemb, negative_pooled_prompt_embeds=npool,
            height=h,width=w,num_inference_steps=steps,guidance_scale=guidance,
            clip_skip=clip_skip,generator=gen).images[0]
    if hires:
        flush()
        hw, hh = (int(dim * hires_scale) // 8 * 8 for dim in (w, h))
        if any(c != [0]*3+[1.0]*2 for c in adjust.values()):
            img = color_balance(img, adjust)
        img_h = img.resize((hw, hh))
        img = refiner(
            prompt_embeds=emb, pooled_prompt_embeds=pool, 
            negative_prompt_embeds=nemb, negative_pooled_prompt_embeds=npool,
            num_inference_steps=hires_steps,guidance_scale=guidance_h,strength=denoise,
            clip_skip=clip_skip,image=img_h,generator=gen_h).images[0]
    display(img.resize((int(w*disp),int(h*disp)),Image.Resampling.LANCZOS))
    img.save(f"{idir}{i:05d}_{seed}.png", pnginfo=meta)
    gc.collect()
    torch.cuda.empty_cache()
del pipe,refiner,emb,nemb,pool,npool
torch.cuda.empty_cache()
""")

ZIP_TPL = Template(r"""
#@title Image ZIP
import os, zipfile
from tqdm.notebook import tqdm

name="download"

dst="$workpath/working/{name}.zip"
if os.path.exists(dst): os.remove(dst)
paths = [os.path.join(r, f) for r, _, fs in os.walk("$workpath/working/t2i_images/") for f in fs]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for p in tqdm(paths, desc="Zipping..."): z.write(p, os.path.join(name, os.path.relpath(p, "$workpath/working/t2i_images/")))
print("Done!")
""")

# ---------- public APIs (use planit() to produce res, final) ----------
def create_plan(filepath: str, workpath: str, saveas: str, title: str,
                vae: str, CivitAPI: str, HuggingAPI: str, UR: str):
    """Write a .py file: Prelude + planit-emitted cells."""
    _ensure_dirs(os.path.join(workpath,"tmp"), ["models","embeddings","vae"])
    prelude = PRELUDE_TPL.safe_substitute(workpath=workpath, hf_token=HuggingAPI, cv_token=CivitAPI, vae_url=vae)
    res, _ = planit(filepath, workpath)
    with open(saveas, "w", encoding="utf-8") as f:
        f.write(f"#{title}\n\n")
        f.write(prelude)
        f.write("\n".join(res))

def create_plan_ipynb(filepath: str, workpath: str, saveas: str, title: str,
                      vae: str, CivitAPI: str, HuggingAPI: str, UR: str):
    """Write a .ipynb: install + Prelude+plan + upload + t2i + zip."""
    _ensure_dirs(os.path.join(workpath,"tmp"), ["models","embeddings","vae"])
    install = INSTALL_TPL.safe_substitute(workpath=workpath)
    prelude = PRELUDE_TPL.safe_substitute(workpath=workpath, hf_token=HuggingAPI, cv_token=CivitAPI, vae_url=vae)
    res, final = planit(filepath, workpath)
    plan_cell = f"#{title}\n\n" + prelude + "\n".join(res)
    upload = UPLOAD_TPL.safe_substitute(workpath=workpath, final=final, repo=UR)
    t2i_cfg = T2I_CFG_TPL.safe_substitute(workpath=workpath, final=final)
    t2i_run = T2I_RUN_TPL.safe_substitute(workpath=workpath)
    zipc = ZIP_TPL.safe_substitute(workpath=workpath)
    cells = [install, plan_cell, upload, t2i_cfg, t2i_run, zipc]
    with open(saveas, "w", encoding="utf-8") as f:
        f.write(_nb_json(cells))