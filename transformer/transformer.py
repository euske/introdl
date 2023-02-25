def encoder(src):
    # 入力列から2つの辞書を作成する。
    h1 = { key1(x): value1(x) for x in src }
    h2 = { key2(x): value2(x) for x in src }
    memory = (h1,h2)
    return memory

def decoder(memory, target):
    # 2つの辞書を使って出力を生成する。
    (h1,h2) = memory
    v1 = [ h1.get(query1(x)) for x in target ]
    v2 = [ h2.get(query2(x)) for x in target ]
    return ff(v1,v2)

def transformer(input):
    # 入力列を memory に変換する。
    memory = encoder(input)
    # 出力列を初期化する。
    output = [BOS]
    while True:
        # ひとつずつ要素を足していく。
        elem = decoder(memory, output)
        output.append(elem)
        # EOS が来たら終了。
        if elem == EOS: break
    return output

BOS = 0
EOS = 999

def key1(x): return x
def value1(x): return x
def key2(x): return x
def value2(x): return 1
def query1(x): return x
def query2(x): return x+1
def ff(v1,v2):
    x1 = v1[-1]
    x2 = v2[-1]
    if x2 is None:
        return EOS
    else:
        return x1+1

print(transformer([BOS,1,2,3,4,5,EOS])) # [BOS,1,2,3,4,5,EOS]

# 入力列 seq 内の自己注意 (self attention) を求める。
def self_attn(seq):
    h1 = { sa_key1(x): sa_value1(x) for x in seq }
    h2 = { sa_key2(x): sa_value2(x) for x in seq }
    #print(h1,h2)
    a1 = [ h1.get(sa_query1(x)) for x in seq ]
    a2 = [ h2.get(sa_query2(x)) for x in seq ]
    #print(a1,a2)
    return [ aa(y1,y2) for (y1,y2) in zip(a1,a2) ]

def sa_key1(x): return x
def sa_value1(x): return x
def sa_key2(x): return x
def sa_value2(x): return x
def sa_query1(x): return x*2
def sa_query2(x): return x/2
def aa(v1,v2):
    if v1 is None and v2 is None: return 0
    return 1

print(self_attn([BOS,1,2,3,4,5,8,EOS])) # [1,1,1,0,1,0,1,0]

def add_positional(seq):
    return [ i*1000+x for (i, x) in enumerate(seq) ]

print(add_positional([BOS,2,5,7,9,20,EOS])) # [0, 1002, 2005, 3007, 4009, 5020, 6999]

def sa_key1(x): return x // 1000
def sa_value1(x): return x % 1000
def sa_key2(x): return x // 1000
def sa_value2(x): return x % 1000
def sa_query1(x): return x // 1000
def sa_query2(x): return (x // 1000)-1
def aa(v1,v2):
    if v1 != v2: return 0
    return 1

print(self_attn(add_positional([BOS,1,1,5,5,2,EOS]))) # [0, 0, 1, 0, 1, 0, 0]
