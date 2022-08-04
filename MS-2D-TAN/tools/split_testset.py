import json

def split_tacos():
    test_json = json.load(open("data/TACoS/test.json"))
    simple_test_json = {}
    complex_test_json = {}
    for k, v in test_json.items():
        simple_test_json[k] = {'timestamps': [], 'sentences': [], 'fps': v['fps'], 'num_frames': v['num_frames']}
        complex_test_json[k] = {'timestamps': [], 'sentences': [], 'fps': v['fps'], 'num_frames': v['num_frames']}
        for sentence, timestamp in zip(v['sentences'], v['timestamps']):
            is_complex = any([keyword in sentence.lower() for keyword in ['before', 'while', 'then', 'after', 'continue']])
            if is_complex:
                complex_test_json[k]['timestamps'].append(timestamp)
                complex_test_json[k]['sentences'].append(sentence)
            else:
                simple_test_json[k]['timestamps'].append(timestamp)
                simple_test_json[k]['sentences'].append(sentence)
    json.dump(simple_test_json, open('data/TACoS/test_simple.json', 'w'))
    json.dump(complex_test_json, open('data/TACoS/test_complex.json', 'w'))

def split_activitynet():
    test_json = json.load(open("data/ActivityNet/test.json"))
    simple_test_json = {}
    complex_test_json = {}
    for k, v in test_json.items():
        simple_test_json[k] = {'timestamps': [], 'sentences': [], 'duration': v['duration']}
        complex_test_json[k] = {'timestamps': [], 'sentences': [], 'duration': v['duration']}
        for sentence, timestamp in zip(v['sentences'], v['timestamps']):
            is_complex = any([keyword in sentence.lower() for keyword in ['before', 'while', 'then', 'after', 'continue', 'again']])
            if is_complex:
                complex_test_json[k]['timestamps'].append(timestamp)
                complex_test_json[k]['sentences'].append(sentence)
            else:
                simple_test_json[k]['timestamps'].append(timestamp)
                simple_test_json[k]['sentences'].append(sentence)
    json.dump(simple_test_json, open('data/ActivityNet/test_simple.json', 'w'))
    json.dump(complex_test_json, open('data/ActivityNet/test_complex.json', 'w'))

if __name__ == '__main__':
    split_tacos()