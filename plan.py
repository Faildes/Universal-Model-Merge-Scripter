import os
import random
import json

def rnm(n):
    order = list(range(ord("0"),ord("9")+1))+list(range(ord("a"),ord("f")+1))
    randlst = [chr(random.choice(order)) for i in range(n)]
    return ''.join(randlst)
def data_construct(l):
    fe = []
    for data in l:
        u = data.splitlines()
        y = [d+"\n" for d in u]
        form = {
           "cell_type": "code",
           "execution_count": None,
           "id": f"{rnm(8)}-{rnm(4)}-{rnm(4)}-{rnm(4)}-{rnm(12)}",
           "metadata": {
            "editable": True,
            "slideshow": {
             "slide_type": ""
            },
            "tags": []
           },
           "outputs": [],
           "source": y
          }
        fe.append(json.dumps(form))
    return fe

def planit(filepath):
    res = []
    with open(filepath, mode="r+") as f:
        line = f.readline()
        while line:
            t = line.rstrip("\n")
            line = f.readline()
            if t == "":
                res.append("")
                continue
            elif t.startswith("//"):
                r = "#" + t[2:]
            elif t.startswith("+"):
                t = t[1:]
                try:
                    if "https://" not in res[-1] and res[-1] != "":
                        res.append("")
                except:
                    pass
                d = t.replace(",","").split(" ")
                if "%LR" in t:
                    mode = "lora"
                else:
                    mode = "checkpoint"
                if d[0][0] == "_":
                    d[0] = f"TEMP{d[0]}"
                if "/api/" in t:
                    r = f"{d[0]} = old_custom_model(\"{d[1]}\",\"{d[0]}\",1,\"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR\")"
                else:
                    r = f"{d[0]} = custom_model(\"{d[1]}\",\"{d[0]}\",mode=\"{mode}\")"
                if "%LC" in t:
                    r = f"{d[0]} = model(\"{d[0]}\",1)"
            elif t.startswith("CM"):
                t = t[3:]
                if res != []:
                    res.append("")
                d = t.replace("+ ","").replace("+T ","").replace("+S ","").replace("- ","").replace("+D ","").split(" ")
                if t.count("+") == 2:
                    if d[0][0] == "_":
                        d[0] = f"TEMP{d[0]}"
                    if d[1][0] == "_":
                        d[1] = f"TEMP{d[1]}"
                    if d[2][0] == "_":
                        d[2] = f"TEMP{d[2]}"
                    if d[5][0] == "_":
                        d[5] = f"TEMP{d[5]}"
                    if "+T" in t:
                        if not line:
                            r = f"""!python merge.py "TRS" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" --model_2 "{d[2]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[3]} \\
--beta {d[4]} \\
--save_half --prune --save_safetensors --output "{d[5]}"
flush()"""
                            final = d[5]
                        else:
                            r = f"""!python merge.py "TRS" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" --model_2 "{d[2]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[3]} \\
--beta {d[4]} \\
--save_half --prune --save_safetensors --output "{d[5]}"
flush()

{d[5]}=model("{d[5]}",1)"""
                            final = d[5]
                    elif "+S" in t:
                        if not line:
                            r = f"""!python merge.py "ST" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" --model_2 "{d[2]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[3]} \\
--beta {d[4]} \\
--save_half --prune --save_safetensors --output "{d[5]}"
flush()"""
                            final = d[5]
                        else:
                            r = f"""!python merge.py "ST" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" --model_2 "{d[2]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[3]} \\
--beta {d[4]} \\
--save_half --prune --save_safetensors --output "{d[5]}"
flush()

{d[5]}=model("{d[5]}",1)"""
                            final = d[5]
                elif "-" in t:
                    if not line:
                        r = f"""!python merge.py "AD" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" --model_2 "{d[2]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[3]} \\
--save_half --prune --save_safetensors --output "{d[4]}"
flush()"""
                        final = d[4]
                    else:
                        r = f"""!python merge.py "AD" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" --model_2 "{d[2]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[3]} \\
--save_half --prune --save_safetensors --output "{d[4]}"
flush()

{d[4]}=model("{d[4]}",1)"""
                        final = d[4]
                elif "+D" in t:
                    if "," in d[2]:
                        j = d[2].split(",",1)
                        j[0] = "\"0"
                        d[2] = ",".join(j)
                    else:
                        j = ["\"0"] + ([d[2]] * 25)
                        d[2] = ",".join(j)
                    if not d[2].endswith("\""):
                        d[2] += "\""
                    if d[0][0] == "_":
                        d[0] = f"TEMP{d[0]}"
                    if d[1][0] == "_":
                        d[1] = f"TEMP{d[1]}"
                    if d[3][0] == "_":
                        d[3] = f"TEMP{d[3]}"
                    if not line:
                        r = f"""!python merge.py "DARE" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[2]} \\
--save_half --prune --save_safetensors --output "{d[3]}"
flush()"""
                        final = d[3]
                    else:
                        r = f"""!python merge.py "DARE" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[2]} \\
--save_half --prune --save_safetensors --output "{d[3]}"
flush()

{d[3]}=model("{d[3]}",1)"""
                        final = d[3]
                else:
                    if d[0][0] == "_":
                        d[0] = f"TEMP{d[0]}"
                    if d[1][0] == "_":
                        d[1] = f"TEMP{d[1]}"
                    if d[3][0] == "_":
                        d[3] = f"TEMP{d[3]}"
                    if not line:
                        r = f"""!python merge.py "WS" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[2]} \\
--save_half --prune --save_safetensors --output "{d[3]}"
flush()"""
                        final = d[3]
                    else:
                        r = f"""!python merge.py "WS" "/kaggle/tmp/models/" "{d[0]}.safetensors" "{d[1]}.safetensors" \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--alpha {d[2]} \\
--save_half --prune --save_safetensors --output "{d[3]}"
flush()

{d[3]}=model("{d[3]}",1)"""
                        final = d[3]
            elif t.startswith("LB"):
                t = t[3:]
                d=t.split(" ")
                if d[0][0] == "_":
                    d[0] = f"TEMP{d[0]}"
                if d[2][0] == "_":
                    d[2] = f"TEMP{d[2]}"
                e = d[1].split(",")
                er=[]
                for q in e:
                    er.append(q.replace(":",".safetensors:"))
                fe = ",".join(er)
                if not line:
                    r = f"""!python lora_bake.py "/kaggle/tmp/models/" "{d[0]}.safetensors" \\
"{fe}" \\
--save_safetensors --output "{d[2]}"
flush()"""
                    final = d[2]
                else:
                    r = f"""!python lora_bake.py "/kaggle/tmp/models/" "{d[0]}.safetensors" \\
"{fe}" \\
--save_safetensors --output "{d[2]}"
flush()

{d[2]}=model("{d[2]}",1)"""
                    final = d[2]
            elif t.startswith("PR"):
                t = t[3:]
                d = t.split(" ")
                if d[0][0] == "_":
                    d[0] = f"TEMP{d[0]}"
                if d[1][0] == "_":
                    d[1] = f"TEMP{d[1]}"
                if not line:
                    r = f"""!python merge.py "NoIn" "/kaggle/tmp/models/" "{d[0]}.safetensors" None \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--save_half --prune --save_safetensors --output "{d[1]}"
flush()"""
                    final = d[1]
                else:
                    r = f"""!python merge.py "NoIn" "/kaggle/tmp/models/" "{d[0]}.safetensors" None \\
--vae "/kaggle/tmp/vae/VAE.safetensors" \\
--save_half --prune --save_safetensors --output "{d[1]}"
flush()

{d[1]}=model("{d[1]}",1)"""
                    final = d[1]
            elif t.startswith("-"):
                try:
                    if "remove_model" not in res[-1] and res[-1] != "":
                        res.append("")
                except:
                    pass
                d = t.replace("-","")
                if d[0] == "_":
                    d = f"TEMP{d}"
                r = f"remove_model({d})"
            res.append(r)
    return res, final

def create_plan(filepath, saveas, title, vae, CivitAPI, HuggingAPI, UR):
    res = []
    pre = r"""from fake_useragent import UserAgent
import os
import shutil
from huggingface_hub import upload_file
import json
import os
import filelock, json, hashlib
import re
import requests
import gc, torch

pref = {
    "format": "SafeTensor",
    "size": "pruned",
    "fp": "fp16"}

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  !pip cache purge
 
def remove_model(path):
    print(f"Delete {os.path.basename(path)}")
    os.remove(path)
    total, used, free = shutil.disk_usage("/")
    print(f"Remain Storage: {free / (2**30):.2f}GB/{total / (2**30):.2f}GB")
    
def make_pref(p,mode):
    pref_set = {
        "size": ["full","pruned"],
        "fp": ["fp16","bf16","fp32"],
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
        r=[[0,0,0],[1, 0, 0],[0, 1, 0],[1, 1, 0],[0, 2, 0],[1, 2, 0],[0, 0, 1],[1, 0, 1],[0, 1, 1],[1, 1, 1],[0, 2, 1],[1, 2, 1]]
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
cache_filename = os.path.join("/kaggle/tmp/models", "cache.json")
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

"""+f"""HFToken = "{HuggingAPI}"
CVToken = "{CivitAPI}"
"""+r"""if not os.path.exists("/kaggle/tmp"):
  os.mkdir("/kaggle/tmp")
if not os.path.exists("/kaggle/tmp/models"):
  os.mkdir("/kaggle/tmp/models")
if not os.path.exists("/kaggle/tmp/embeddings"):
  os.mkdir("/kaggle/tmp/embeddings")
if not os.path.exists("/kaggle/tmp/vae"):
  os.mkdir("/kaggle/tmp/vae")
  
%cd /kaggle/tmp/
user_header = f"\"Authorization: Bearer {HFToken}\""
model_path = "/kaggle/tmp/models/"

def model(name,format=0):
    ext = "ckpt" if format == 0 else "safetensors"
    sha256_set(f"{model_path}{name}.{ext}", f"checkpoint/{name}", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR")
    return f"{model_path}{name}.{ext}"
  
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
  if os.path.exists(f"/kaggle/tmp/models/{checkpoint_name}.{ext}"):
    return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/models/ -o {checkpoint_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/models/" -o {checkpoint_name}.{ext}
  if sha256 is not None:
    sha256_set(f"{model_path}{checkpoint_name}.{ext}", f"{mode}/{checkpoint_name}", sha256)
  return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"

def old_custom_model(url, checkpoint_name=None, format=0, sha256=None):
  user_token = HFToken if "huggingface" in url else CVToken
  if format == 0:
    ext = "ckpt"
  elif format == 1:
    ext = "safetensors"
  if os.path.exists(f"/kaggle/tmp/models/{checkpoint_name}.{ext}"):
    return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/models/ -o {checkpoint_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/models/" -o {checkpoint_name}.{ext}
  if sha256 is not None:
    sha256_set(f"{model_path}{checkpoint_name}.{ext}", f"checkpoint/{checkpoint_name}", sha256)
  return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"
  
def custom_vae(url, vae_name, format=0):
  user_token = HFToken if "huggingface" in url else CVToken
  ext = ""
  if format == 0:
    ext = "pt"
  elif format == 1:
    ext = "safetensors"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/vae/ -o {vae_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/vae/" -o {vae_name}.{ext}
    return f"/kaggle/tmp/vae/{vae_name}.{ext}"

"""+f"""custom_vae("{vae}","VAE",1)

%cd /kaggle/working/merge-models

"""
    res, _ = planit(filepath)
    with open(saveas, mode="a+") as f:
        f.write(f"#{title}\n\n")
        f.write(pre)
        f.write("\n".join(res))

def create_plan_ipynb(filepath, saveas, title, vae, CivitAPI, HuggingAPI,UR):
    dp = ["""!pip install compel lora torch safetensors accelerate fake_useragent diffusers["torch"] transformers torchsde ninja xformers git+https://github.com/huggingface/diffusers
!pip install -U peft transformers
!apt-get -y install -qq aria2
%cd /kaggle/working/
!git clone https://github.com/Faildes/merge-models"""]
    res = []
    pre = r"""from fake_useragent import UserAgent
import os
import shutil
from huggingface_hub import upload_file
import json
import os
import filelock, json, hashlib
import re
import requests
import gc, torch

pref = {
    "format": "SafeTensor",
    "size": "pruned",
    "fp": "fp16"}

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  !pip cache purge

def remove_model(path):
    print(f"Delete {os.path.basename(path)}")
    os.remove(path)
    total, used, free = shutil.disk_usage("/")
    print(f"Remain Storage: {free / (2**30):.2f}GB/{total / (2**30):.2f}GB")
    
def make_pref(p,mode):
    pref_set = {
        "size": ["full","pruned"],
        "fp": ["fp16","bf16","fp32"],
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
        r=[[0,0,0],[1, 0, 0],[0, 1, 0],[1, 1, 0],[0, 2, 0],[1, 2, 0],[0, 0, 1],[1, 0, 1],[0, 1, 1],[1, 1, 1],[0, 2, 1],[1, 2, 1]]
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
cache_filename = os.path.join("/kaggle/tmp/models", "cache.json")
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

"""+f"""HFToken = "{HuggingAPI}"
CVToken = "{CivitAPI}"
"""+r"""if not os.path.exists("/kaggle/tmp"):
  os.mkdir("/kaggle/tmp")
if not os.path.exists("/kaggle/tmp/models"):
  os.mkdir("/kaggle/tmp/models")
if not os.path.exists("/kaggle/tmp/embeddings"):
  os.mkdir("/kaggle/tmp/embeddings")
if not os.path.exists("/kaggle/tmp/vae"):
  os.mkdir("/kaggle/tmp/vae")
  
%cd /kaggle/tmp/
user_header = f"\"Authorization: Bearer {HFToken}\""
model_path = "/kaggle/tmp/models/"

def model(name,format=0):
    ext = "ckpt" if format == 0 else "safetensors"
    sha256_set(f"{model_path}{name}.{ext}", f"checkpoint/{name}", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR")
    return f"{model_path}{name}.{ext}"
  
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
  if os.path.exists(f"/kaggle/tmp/models/{checkpoint_name}.{ext}"):
    return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/models/ -o {checkpoint_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/models/" -o {checkpoint_name}.{ext}
  if sha256 is not None:
    sha256_set(f"{model_path}{checkpoint_name}.{ext}", f"{mode}/{checkpoint_name}", sha256)
  return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"

def old_custom_model(url, checkpoint_name=None, format=0, sha256=None):
  user_token = HFToken if "huggingface" in url else CVToken
  if format == 0:
    ext = "ckpt"
  elif format == 1:
    ext = "safetensors"
  if os.path.exists(f"/kaggle/tmp/models/{checkpoint_name}.{ext}"):
    return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/models/ -o {checkpoint_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/models/" -o {checkpoint_name}.{ext}
  if sha256 is not None:
    sha256_set(f"{model_path}{checkpoint_name}.{ext}", f"checkpoint/{checkpoint_name}", sha256)
  return f"/kaggle/tmp/models/{checkpoint_name}.{ext}"
  
def custom_vae(url, vae_name, format=0):
  user_token = HFToken if "huggingface" in url else CVToken
  ext = ""
  if format == 0:
    ext = "pt"
  elif format == 1:
    ext = "safetensors"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/vae/ -o {vae_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/vae/" -o {vae_name}.{ext}
    return f"/kaggle/tmp/vae/{vae_name}.{ext}"

"""+f"""custom_vae("{vae}","VAE",1)

%cd /kaggle/working/merge-models

"""
    res, final = planit(filepath)
    nwx = f"#{title}\n\n"
    nwx += pre
    nwx += "\n".join(res)
    dp.append(nwx)
    dp.append(f"""from huggingface_hub import upload_file
User_Repository = "{UR}"
%cd /kaggle/tmp/models
upload_file(path_or_fileobj="/kaggle/tmp/models/{final}.safetensors", 
            path_in_repo="{final}.safetensors", 
            repo_id=User_Repository, 
            token=HFToken)
!pip cache purge""")
    dp.append("""#@title Pipe Config
#@markdown Play this after putting informations.
import requests
from fake_useragent import UserAgent
import torch
import os
import datetime
import gc
import safetensors.torch
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import diffusers
import filelock, json, hashlib
import re
pref = {
    "format": "SafeTensor",
    "size": "pruned",
    "fp": "fp16"}

def get_dl(url, version:str =None):
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
            stored = []
            for a in model["files"]:
                meta = a["metadata"]
                if meta != pref: continue
                dllink=a["downloadUrl"]
                sha256=a["hashes"]["SHA256"]
                ext = a["metadata"]["format"]
                if ext == "SafeTensor":
                    ex = 1
                else:
                    ex = 0
                break
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
if not os.path.exists("/kaggle/tmp"):
  os.mkdir("/kaggle/tmp")
if not os.path.exists("/kaggle/tmp/models"):
  os.mkdir("/kaggle/tmp/models")
if not os.path.exists("/kaggle/tmp/embeddings"):
  os.mkdir("/kaggle/tmp/embeddings")
if not os.path.exists("/kaggle/tmp/vae"):
  os.mkdir("/kaggle/tmp/vae")
SCHEDULERS = {
    "unipc": [diffusers.schedulers.UniPCMultistepScheduler,{},"UniPC"],
    "euler_a": [diffusers.schedulers.EulerAncestralDiscreteScheduler,{}, "Euler a"],
    "euler": [diffusers.schedulers.EulerDiscreteScheduler,{}, "Euler"],
    "ddim": [diffusers.schedulers.DDIMScheduler,{},"DDIM"],
    "ddpm": [diffusers.schedulers.DDPMScheduler,{},"DDPM"],
    "deis": [diffusers.schedulers.DEISMultistepScheduler,{},"DEIS"],
    "dpm2": [diffusers.schedulers.KDPM2DiscreteScheduler,{},"DPM2"],
    "dpm2_karras": [diffusers.schedulers.KDPM2DiscreteScheduler,{"use_karras_sigmas":True},"DPM2 Karras"],
    "dpm2-a": [diffusers.schedulers.KDPM2AncestralDiscreteScheduler,{},"DPM2 a"],
    "dpm2-a_karras": [diffusers.schedulers.KDPM2AncestralDiscreteScheduler,{"use_karras_sigmas":True},"DPM2 a Karras"],
    "dpm++_2s_a": [diffusers.schedulers.DPMSolverSinglestepScheduler,{},"DPM++ 2S a"],
    "dpm++_2s_a_karras": [diffusers.schedulers.DPMSolverSinglestepScheduler,{"use_karras_sigmas":True},"DPM++ 2S a Karras"],
    "dpm++_2m": [diffusers.schedulers.DPMSolverMultistepScheduler,{},"DPM++ 2M"],
    "dpm++_2m_karras": [diffusers.schedulers.DPMSolverMultistepScheduler,{"use_karras_sigmas":True},"DPM++ 2M Karras"],
    "dpm++_2m_sde": [diffusers.schedulers.DPMSolverMultistepScheduler,{"algorithm_type":"sde-dpmsolver++"},"DPM++ 2M SDE"],
    "dpm++_2m_sde_karras": [diffusers.schedulers.DPMSolverMultistepScheduler,{"algorithm_type":"sde-dpmsolver++","use_karras_sigmas": True},"DPM++ 2M SDE Karras"],
    "dpm++_sde": [diffusers.schedulers.DPMSolverSDEScheduler,{},"DPM++ SDE"],
    "dpm++_sde_karras": [diffusers.schedulers.DPMSolverSDEScheduler,{"use_karras_sigmas":True},"DPM++ SDE Karras"],
    "heun": [diffusers.schedulers.HeunDiscreteScheduler,{},"Heun"],
    "heun_karras": [diffusers.schedulers.HeunDiscreteScheduler,{"use_karras_sigmas":True},"Heun Karras"],
    "lms": [diffusers.schedulers.LMSDiscreteScheduler,{},"LMS"],
    "lms_karras": [diffusers.schedulers.LMSDiscreteScheduler,{"use_karras_sigmas":True},"LMS Karras"],
    "pndm": [diffusers.schedulers.PNDMScheduler,{},"PNDM"],
}
cache_filename = os.path.join("/kaggle/tmp/", "cache.json")
cache_data = None
scheduler = "choose from below list" #@param ["unipc", "euler_a", "euler", "ddim", "ddpm", "deis", "dpm2", "dpm2_karras", "dpm2-a", "dpm2-a_karras", "dpm++_2s_a", "dpm++_2s_a_karras", "dpm++_2m", "dpm++_2m_karras", "dpm++_2m_sde", "dpm++_2m_sde_karras", "dpm++_sde", "dpm++_sde_karras", "heun", "heun_karras", "lms", "lms_karras", "pndm"]
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
def flush():
  gc.collect()
  !pip cache purge
  torch.cuda.empty_cache()
def custom_model(url, name, format=0, loc=False, s256=None):
  user_token = HFToken if "huggingface" in url else CVToken
  ext = ""
  if format == 0:
    ext = "ckpt"
  elif format == 1:
    ext = "safetensors"
  if not os.path.exists(f"/kaggle/tmp/models/{name}.{ext}"):
      if "huggingface" in url:
        user_header = f"\"Authorization: Bearer {user_token}\""
        !aria2c --console-log-level=error --header={user_header} -c -x 16 -s 16 -k 1M {url} -d /kaggle/tmp/models/ -o {name}.{ext}
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
        !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "/kaggle/tmp/models/" -o {name}.{ext}
      if s256 is not None:
        sha256_set(f"/kaggle/tmp/models/{name}.{ext}", name, s256)
  if loc:
    s256 = sha256(f"/kaggle/tmp/models/{name}.{ext}", name)
  print(s256)
  return [f"/kaggle/tmp/models/{name}.{ext}",s256]
def custom_embed(url, embed_name, format=0):
  user_token = HFToken if "huggingface" in url else CVToken
  if format == 0:
    ext = "pt"
  elif format == 1:
    ext = "safetensors"
  if "safetensors" in url:
    ext = "safetensors"
  elif "pt" in url:
    ext = "pt"
  if "huggingface" in url:
    user_header = f"\"Authorization: Bearer {user_token}\""
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M --header={user_header} "{url}" -d /kaggle/tmp/embeddings/ -o {embed_name}.{ext}
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
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d /kaggle/tmp/embeddings/ -o {embed_name}.{ext}
  return f"/kaggle/tmp/embeddings/{embed_name}.{ext}"
#@markdown Choose the models you want
from safetensors import safe_open
from safetensors.torch import save_file

def fix_diffusers_model_conversion(load_path: str, save_path: str):
    if not os.path.exists(save_path):
      # load original
      tensors = {}
      with safe_open(load_path, framework="pt") as f:
          for key in f.keys():
              tensors[key] = f.get_tensor(key)

      # migrate
      new_tensors = {}
      for k, v in tensors.items():
          new_key = k
          # only fix the vae
          if 'first_stage_model.' in k:
              # migrate q, k, v keys
              new_key = new_key.replace('.to_q.weight', '.q.weight')
              new_key = new_key.replace('.to_q.bias', '.q.bias')
              new_key = new_key.replace('.to_k.weight', '.k.weight')
              new_key = new_key.replace('.to_k.bias', '.k.bias')
              new_key = new_key.replace('.to_v.weight', '.v.weight')
              new_key = new_key.replace('.to_v.bias', '.v.bias')
          new_tensors[new_key] = v

      # save
      save_file(new_tensors, save_path)
checkpoint = "Output"
ext = "safetensors"
cpath = "/kaggle/temp/models/"+checkpoint+"."+ext
chash = sha256(cpath, checkpoint)
try:
  scd_changed = scd_name == SCHEDULERS[scheduler][2]
except:
  pass
try:
  pipe=StableDiffusionXLPipeline.from_single_file(cpath, torch_dtype=torch.float16, scheduler=scd, use_safetensors=True, variant="fp16")
  assert scd_changed
except:
  pipe = StableDiffusionXLPipeline.from_single_file(cpath, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
  scd = SCHEDULERS[scheduler][0].from_config(pipe.scheduler.config, **SCHEDULERS[scheduler][1])
  scd_name = SCHEDULERS[scheduler][2]
  pipe=StableDiffusionXLPipeline.from_single_file(cpath, torch_dtype=torch.float16, scheduler=scd, use_safetensors=True, variant="fp16")
pipe.safety_checker = None
pipe = pipe.to("cuda:0")
flush()

#@markdown Choose the alpha of LoRAs you want
novasphere = True
modelpath = "/kaggle/tmp/models/"
lpath = {}
if novasphere:
  lpath["novasphere"] = [custom_model("https://civitai.com/models/439098/nova-sphere-style","novasphere",1),modelpath,""]

#@markdown Choose the Embeddings you want
negativexl = True 
aissist = True 

epath = {}
if negativexl:
  embed = "NegativeXL"
  embed_url = "https://civitai.com/api/download/models/134583?type=Model&format=SafeTensor"
  epath[embed]=custom_embed(embed_url,embed)
if aissist:
  embed = "AIssist"
  embed_url = "https://civitai.com/api/download/models/403492?type=Model&format=SafeTensor"
  epath[embed]=custom_embed(embed_url,embed)
from safetensors.torch import load_file

init_pipe = pipe""")
    dp.append("""#@title t2i
import torch
import os
import datetime
import gc
import safetensors.torch
import re
from IPython.display import display
import random
import copy
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from PIL.PngImagePlugin import PngInfo
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from accelerate import PartialState
import numpy
pipe = init_pipe

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
        
def lora_prompt(prompt, pipe, lhash):
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
        except:
          pass
    pipe.set_adapters(adap_list, adapter_weights=alphas)
    #refine.set_adapters(adap_list, adapter_weights=alphas)
    return prompt, lhash
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  !pip cache purge

def bpro(prompt):
    k = prompt.split(",")
    thu = []
    for g in k:
        f = g.count(" ")
        thu.append([g, f+1])
    off = 0
    nl = []
    t = 0
    for x in thu:
        if "BREAK" in x[0]:
            tok = t+off
            add = tok % 75
            nl += [" "]*add
            off += add
            continue
        t += x[1]
        nl.append(x[0])
    return ",".join(nl)

if not os.path.exists("/kaggle/working/t2i_images"):
  os.mkdir("/kaggle/working/t2i_images")

mdir = "/kaggle/tmp/models/"
idir = "/kaggle/working/t2i_images/"
prompt = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, realistic, photo, dynamic angle, dramatic shadows, high quality, BREAK, cyberpunk, blue lights, cinematic portrait photo, young woman with (shoulder-length)0.5 brunette hair and hazel eyes, wearing a black formfitting high-tech futuristic outfit and pants"
neg = "blurry, signature, username, watermark, jpeg artifacts, normal quality, worst quality, low quality, missing fingers, extra digits, fewer digits, bad eye" #@param {type:"string"}

w=768 #@param {type:"slider", min:512, max:2048, step:128}
h=1280 #@param {type:"slider", min:512, max:2048, step:128}
hires_steps=40 #@param {type:"slider", min:10, max:100, step:1}
hires_scale = 1.5 #@param {type:"slider", min:1.0, max:4.0, step:0.1}
refine=True
global_seed=-1 #@param
global_hires_seed=-2 #@param
steps=40 #@param {type:"slider", min:10, max:50, step:1}
guidance=6 #@param {type:"slider", min:0.5, max:15.0, step:0.5}
denoise=0.6 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
clip_skip = 1 #@param {type:"slider", min:1, max:12, step:1}
num_gen = 1 #@param {type:"slider", min:1, max:4, step:1}
num_rp = 1 #@param {type:"slider", min:1, max:2, step:1}
rand_seed = 0
copy_seed = False
if global_seed == -1: rand_seed += 1
if global_hires_seed == -1: rand_seed += 2
if global_hires_seed == -2: copy_seed = True
lhash = {}
pp, lhash = lora_prompt(prompt, pipe, lhash)
pp = bpro(pp)
np = bpro(neg)

compel_proc = Compel(
    tokenizer=[pipe.tokenizer,pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder,pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    truncate_long_prompts=False,
    device="cuda:0")


embeds, pooled = compel_proc.build_conditioning_tensor(pp)
negative_embeds, neg_pooled = compel_proc.build_conditioning_tensor(np)

[embeds, negative_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([embeds, negative_embeds])

device = "cpu"
i = 0
disp_size = 512 / min(w, h)
global_seed = random.randrange(4294967294) if rand_seed >= 1 else global_seed
global_hires_seed = random.randrange(4294967294) if rand_seed >= 2 else global_hires_seed
if copy_seed :global_hires_seed = global_seed
print(global_seed)
if global_hires_seed != global_seed: print(global_hires_seed)
numpy.random.seed(global_seed)
seeds=numpy.random.randint(0, 4294967294, num_gen)
numpy.random.seed(global_hires_seed)
hires_seeds=numpy.random.randint(0, 4294967294, num_gen)
while i < num_gen:
  seed = int(seeds[i])
  hires_seed = int(hires_seeds[i])
  geninfo = f"{prompt}\\nNegative prompt: {neg}\\nSteps: {steps}, Sampler: {scd_name}, CFG scale: {guidance}, Global Seed: {global_seed}, Seed: {seed}, Size: {w}x{h}, Clip skip: {clip_skip}, Model hash: {chash}, Model: {checkpoint}"
  if len(lhash) > 0:
    geninfo += ", Lora hashes: \""
    n = ""
    for q, u in lhash.items():
      n += f"{q}: {u}, "
    n = n[:-2]
    geninfo += f"{n}\""
  metadata = PngInfo()
  metadata.add_text("parameters", geninfo)
  generator = torch.Generator(device).manual_seed(seed)
  with torch.inference_mode():
      image = pipe(
              prompt_embeds=embeds, 
              pooled_prompt_embeds=pooled, 
              negative_prompt_embeds=negative_embeds, 
              negative_pooled_prompt_embeds=neg_pooled, 
              height=h, width=w, 
              num_inference_steps=steps, 
              guidance_scale=guidance,
              generator=generator).images[0]
  flush()
  display(image.resize((int(w*disp_size),int(h*disp_size))))
  image.save(f"{idir}{i:05d}_{global_seed}.png", pnginfo=metadata)
  i += 1
flush()
del pipe
torch.device("cpu")
torch.cuda.empty_cache()
torch.device("cuda:0")
torch.cuda.empty_cache()""")
    tr = data_construct(dp)
    with open(saveas, mode="w+") as f:
        f.write("""{"cells":[""")
        f.write(",".join(tr))
        f.write("""],"metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)","language": "python","name": "python3"},"language_info": {"codemirror_mode": {"name": "ipython","version": 3},"file_extension": ".py","mimetype": "text/x-python","name": "python","nbconvert_exporter": "python","pygments_lexer": "ipython3","version": "3.10.6"}},"nbformat": 4,"nbformat_minor": 5}""")
