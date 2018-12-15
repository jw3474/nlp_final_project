PUNC_MAP = {'&apos;': '\'', '&quot;': '\"'}
SPECIAL_TOKENS = ['<pad>', '<eos>', '<start>', '<unk>']

def convert_punc(t):
    """
    Handle single and double quotes properly.
    For example, 's -> [', s], 've ->[', have]
    """
    m = PUNC_MAP
    ret = [t]
    for k in m:
        if k in t:
            t = t.replace(k, m[k])
            if len(t) > 0:
                ret = [t[0]]
                if t[1:] == 've':
                    ret.append('have')
                elif t[1:] == 'll':
                    ret.append('will')
                elif t[1:] == 'd':
                    ret.append('should')
                elif t[1:] == 're':
                    ret.append('are')
                elif t[1:] == 'm':
                    ret.append('am')
                else:
                    ret.append(t[1:])
            else:
                ret = [t]
            break
    return ret