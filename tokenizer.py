text = '''
Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” 
in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus 
its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still 
finding the whole thing mysterious, even 30 years after Unicode’s inception.
'''

tokens = text.encode("utf-8") # raw bytes 
tokens = list(map(int, tokens)) # convert into a list of integers in range 0-255
print("-----")
# print(text)
print("-----")
print("length of text = ", len(text))
print("-----")
# print(tokens)
print("length of tokens = ", len(tokens))

def get_pairs(ids):
    count = {}
    for pair in zip(ids, ids[1:]): #iterating through consecutive integers through the list(check diary for explanation)
        count[pair] = count.get(pair, 0) + 1 #frequency DSA

    return count

stats = get_pairs(tokens)
# print(stats)
# print(sorted(((v,k) for k,v in stats.items()), reverse=True))
'''
# iterating through the items of the dictionary, like for example -> ab is the pair and the utf values are 103 and 203. 
# so these(103, 203) are the keys and the value will be the frequency of that pair. (like how many times ab has occurred in the text)
# so what we are doing here is, printing the values of the keys and then sorting them in descending order so that we will have the 
# highest frequency pairs at the first. 

# in the output, we have [20, (101, 32)] as the first element. to check which characters these are : 

print(chr(101))
print(chr(32))

101 is e and 32 is whitespace.
This means that there are 20 words ending with e(becuase whitespace will always come after the letter ends)

'''