import random
import jieba

# 判断是否为'?X'
def is_variable(pat):
    return pat.startswith('?') and all(s.isalpha() for s in pat[1:])

# 判断是否为'?*X'
def is_pattern_segment(pattern):
    return pattern.startswith('?*') and all(a.isalpha() for a in pattern[2:])

# 获得匹配的变量
def pat_match(pattern,saying):
    if not pattern or not saying:return []

    if is_variable(pattern[0]):
        return [(pattern[0],saying[0])] + pat_match(pattern[1:],saying[1:])
    else:
        if pattern[0] != saying[0]:return []
        else:
            return pat_match(pattern[1:],saying[1:])

# print(pat_match("?X greater than ?Y than ?Z".split(), "3 greater than 2 than 4".split()))

# 如果我们知道了每个变量对应的是什么，那么我们就可以很方便的使用我们定义好的模板进行替换：
# 为了方便接下来的替换工作，我们新建立两个函数，
# 一个是把我们解析出来的结果变成一个dictionary，一个是依据这个dictionary依照我们的定义的方式进行替换
def pat_to_dict(patterns):
    return {k:v for k,v in patterns}

def subsitite(rule,parsed_rules):
    if not rule:return []
    return [parsed_rules.get(rule[0],rule[0])] + subsitite(rule[1:],parsed_rules)
    # print(parsed_rules.get(rule[0],rule[0]))  # 此处的输入的parsed_rules是一个字典,
                                                # 若字典中存在{'?X': 'iPhone'}则输出的是 键 rule[0]('?X')的值'iPhone'
defined_patterns = {
    "I need ?X": ["Image you will get ?X soon", "Why do you need ?X ?"],
    "My ?X told me something": ["Talk about more about your ?X", "How do you think about your ?X ?"],
    "I want ?X":["Why do you want ?X", "How do you get ?X"]
}

def get_response(saying, rules):
    for k in rules.keys():
        pattern = pat_match(k.split(),saying.split())
        # print(random.choice(defined_patterns.get(k)))
        rule = random.choice(rules.get(k)).split()
        if pattern:
            return ' '.join(subsitite(rule, pat_to_dict(pattern)))
    return None

# get_response("I need jiwawa",defined_patterns)
print(get_response("I need jiwawa",defined_patterns))
print(get_response("My dog told me something",defined_patterns))
print(get_response("I want jiwawa",defined_patterns))

# 将pat_match修改如下
# 找出所有与'?*X'或'?X'匹配的关键词
def pat_match_with_seg(pattern,saying):
    if not pattern or not saying:return []

    pat = pattern[0]

    if is_variable(pat):
        return [(pat,saying[0])] + pat_match_with_seg(pattern[1:],saying[1:])
    elif is_pattern_segment(pat):
        match,index = segment_match(pattern,saying)
        return [match] + pat_match_with_seg(pattern[1:],saying[index:])
    elif pat == saying[0]:
        return pat_match_with_seg(pattern[1:],saying[1:])
    else:
        return fail

# 这段程序里比较重要的一个新函数是 segment_match，
# 这个函数输入是一个以segment_pattern开头的模式，
# 尽最大可能进行，匹配到这个边长的变量对于的部分。

# 找出 一个与'?*X'或'?X'匹配的关键词元组，
def segment_match(pattern,saying):
    seg_pat,rest = pattern[0],pattern[1:]
    seg_pat = seg_pat.replace('?*','?')

    if not rest:return (seg_pat,saying),len(saying)

    for i,token in enumerate(saying):
        if rest[0] == token and is_match(rest[1:],saying[(i+1):]):
            return (seg_pat,saying[:i]),i

    return (seg_pat,saying),len(saying)

# 在寻找一个与'?*X'匹配的关键词元组判断'?*X'情况下，判断给与的匹配模板(pattern)是否正确
def is_match(rest,saying):
    if not rest and not saying:
        return True
    if not all(a.isalpha() for a in rest[0]):
        return True
    if rest[0] != saying[0]:
        return False
    return is_match(rest[1:],saying[1:])


# 中文问答
rule_responses = {
    '?*x hello ?*y': ['How do you do', 'Please state your problem'],
    '?*x I want ?*y': ['what would it mean if you got ?y', 'Why do you want ?y', 'Suppose you got ?y soon'],
    '?*x if ?*y': ['Do you really think its likely that ?y', 'Do you wish that ?y', 'What do you think about ?y', 'Really-- if ?y'],
    '?*x no ?*y': ['why not?', 'You are being a negative', 'Are you saying \'No\' just to be negative?'],
    '?*x I was ?*y': ['Were you really', 'Perhaps I already knew you were ?y', 'Why do you tell me you were ?y now?'],
    '?*x I feel ?*y': ['Do you often feel ?y ?', 'What other feelings do you have?'],
    '?*x你好?*y': ['你好呀', '请告诉我你的问题'],
    '?*x我想?*y': ['你觉得?y有什么意义呢？', '为什么你想?y', '你可以想想你很快就可以?y了'],
    '?*x我想要?*y': ['?x想问你，你觉得?y有什么意义呢?', '为什么你想?y', '?x觉得... 你可以想想你很快就可以有?y了', '你看?x像?y不', '我看你就像?y'],
    '?*x喜欢?*y': ['喜欢?y的哪里？', '?y有什么好的呢？', '你想要?y吗？'],
    '?*x讨厌?*y': ['?y怎么会那么讨厌呢?', '讨厌?y的哪里？', '?y有什么不好呢？', '你不想要?y吗？'],
    '?*xAI?*y': ['你为什么要提AI的事情？', '你为什么觉得AI要解决你的问题？'],
    '?*x机器人?*y': ['你为什么要提机器人的事情？', '你为什么觉得机器人要解决你的问题？'],
    '?*x对不起?*y': ['不用道歉', '你为什么觉得你需要道歉呢?'],
    '?*x我记得?*y': ['你经常会想起这个吗？', '除了?y你还会想起什么吗？', '你为什么和我提起?y'],
    '?*x如果?*y': ['你真的觉得?y会发生吗？', '你希望?y吗?', '真的吗？如果?y的话', '关于?y你怎么想？'],
    '?*x我?*z梦见?*y':['真的吗? --- ?y', '你在醒着的时候，以前想象过?y吗？', '你以前梦见过?y吗'],
    '?*x妈妈?*y': ['你家里除了?y还有谁?', '嗯嗯，多说一点和你家里有关系的', '她对你影响很大吗？'],
    '?*x爸爸?*y': ['你家里除了?y还有谁?', '嗯嗯，多说一点和你家里有关系的', '他对你影响很大吗？', '每当你想起你爸爸的时候， 你还会想起其他的吗?'],
    '?*x我愿意?*y': ['我可以帮你?y吗？', '你可以解释一下，为什么想?y'],
    '?*x我很难过，因为?*y': ['我听到你这么说， 也很难过', '?y不应该让你这么难过的'],
    '?*x难过?*y': ['我听到你这么说， 也很难过',
                 '不应该让你这么难过的，你觉得你拥有什么，就会不难过?',
                 '你觉得事情变成什么样，你就不难过了?'],
    '?*x就像?*y': ['你觉得?x和?y有什么相似性？', '?x和?y真的有关系吗？', '怎么说？'],
    '?*x和?*y都?*z': ['你觉得?z有什么问题吗?', '?z会对你有什么影响呢?'],
    '?*x和?*y一样?*z': ['你觉得?z有什么问题吗?', '?z会对你有什么影响呢?'],
    '?*x我是?*y': ['真的吗？', '?x想告诉你，或许我早就知道你是?y', '你为什么现在才告诉我你是?y'],
    '?*x我是?*y吗': ['如果你是?y会怎么样呢？', '你觉得你是?y吗', '如果你是?y，那一位着什么?'],
    '?*x你是?*y吗':  ['你为什么会对我是不是?y感兴趣?', '那你希望我是?y吗', '你要是喜欢， 我就会是?y'],
    '?*x你是?*y' : ['为什么你觉得我是?y'],
    '?*x因为?*y' : ['?y是真正的原因吗？', '你觉得会有其他原因吗?'],
    '?*x我不能?*y': ['你或许现在就能?*y', '如果你能?*y,会怎样呢？'],
    '?*x我觉得?*y': ['你经常这样感觉吗？', '除了到这个，你还有什么其他的感觉吗？'],
    '?*x我?*y你?*z': ['其实很有可能我们互相?y'],
    '?*x你为什么不?*y': ['你自己为什么不?y', '你觉得我不会?y', '等我心情好了，我就?y'],
    '?*x好的?*y': ['好的', '你是一个很正能量的人'],
    '?*x嗯嗯?*y': ['好的', '你是一个很正能量的人'],
    '?*x不嘛?*y': ['为什么不？', '你有一点负能量', '你说 不，是想表达不想的意思吗？'],
    '?*x不要?*y': ['为什么不？', '你有一点负能量', '你说 不，是想表达不想的意思吗？'],
    '?*x有些人?*y': ['具体是哪些人呢?'],
    '?*x有的人?*y': ['具体是哪些人呢?'],
    '?*x某些人?*y': ['具体是哪些人呢?'],
    '?*x每个人?*y': ['我确定不是人人都是', '你能想到一点特殊情况吗？', '例如谁？', '你看到的其实只是一小部分人'],
    '?*x所有人?*y': ['我确定不是人人都是', '你能想到一点特殊情况吗？', '例如谁？', '你看到的其实只是一小部分人'],
    '?*x总是?*y': ['你能想到一些其他情况吗?', '例如什么时候?', '你具体是说哪一次？', '真的---总是吗？'],
    '?*x一直?*y': ['你能想到一些其他情况吗?', '例如什么时候?', '你具体是说哪一次？', '真的---总是吗？'],
    '?*x或许?*y': ['你看起来不太确定'],
    '?*x可能?*y': ['你看起来不太确定'],
    '?*x他们是?*y吗？': ['你觉得他们可能不是?y？'],
    '?*x': ['很有趣', '请继续', '我不太确定我很理解你说的, 能稍微详细解释一下吗?']
}

def is_chinese(chenk_string):
    for s in chenk_string:
        if u'\u4e00' <= s <= u'\u9fff':
            return True
        return False

# print(is_chinese("中国"))
# print(is_chinese("X中国"))
# print(is_chinese("jiwawa")

def get_pattern(pattern):
    pattern = pattern.replace('?*x','xxx').replace('?*y','yyy').replace('?*z','zzz').replace('?x','xx').replace('?y','yy')
    match = ','.join(jieba.cut(pattern))
    match = match.replace('xxx','?*x').replace('yyy','?*y').replace('?z','zz').replace('xx','?x').replace('yy','?y')
    # print(match)
    return match.split(',')

def get_response_chinese(saying,response_rules):
    if not is_chinese(saying):
        for key in response_rules.keys():
            English_pattern = pat_match_with_seg(key.split(), saying.split())
            rule_eng = random.choice(response_rules.get(key)).split()
            if len(English_pattern[0][1]) != len(saying.split()): # 此处不太清楚如何区分不同的规则
                if English_pattern:
                    return ' '.join(subsitite(rule_eng, pat_to_dict(English_pattern)))
    else:
        for key in response_rules.keys():
            Chinese_pattern = pat_match_with_seg(get_pattern(key), list(jieba.cut(saying)))
            rule_ch = random.choice(response_rules.get(key)).split()
            if len(Chinese_pattern[0][1]) != len(saying.split()):
                if Chinese_pattern:
                    return ' '.join(subsitite(rule_ch, pat_to_dict(Chinese_pattern)))
    return None

# print(get_response_chinese('jiwawa I want books',rule_responses))
# print(get_response_chinese('jiwawa hello jiwawa',rule_responses))

print(get_response_chinese('人工智能',rule_responses))


"""
问题4
(1)这样的程序有什么优点？有什么缺点？你有什么可以改进的方法吗？
优点：在匹配模式以内的语句，匹配比较准确，回复符合预设的逻辑
缺点：进行匹配的模式必须人为事先预先定义，如果出现一个没有定义过的模式，那么返回的结果很差或不能返回结果
     匹配的模型比较相似时，也无法选择更优的匹配模式
改进方法：预先定义好一个语料库或字典，对输入的内容先进行预处理后，再进行输入。

(2)什么是数据驱动？数据驱动在这个程序里如何体现？
我认为数据驱动就是：预先定义好一个程序，当输入不同时，程序能有效的处理不同的输入内容，回复相应的结果，而无须修改程序。
每次变化的是输入的数据，而程序本身不用再改动

(3)数据驱动与AI的关系是什么？
我认为数据驱动与AI是相似的。现在的AI都是基于数据的，人们预处理好数据，然后将数据送入模型进行训练，在训练过程中无人为的干涉。
也就是说对于AI来说也是一样，每次输入的数据可能不同，但是作为程序的模型无需改动，只是可能由于输入的数据不用，最好得到的结果不用。
"""