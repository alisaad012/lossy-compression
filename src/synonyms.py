thesaurus = {}

def setup():
    with open('mobythes.aur', 'r') as f_in:
        lines = [set([word for word in line.split(',') if ' ' not in word]) for line in f_in.read().split('\n')]
    for line in lines:
        for word in line:
            global thesaurus
            thesaurus[word] = line

def clean(string):
    letters = []
    for char in string:
        if char.isalpha():
            letters.append(char.lower())
    return ''.join(letters)

def is_synonym(word1, word2):
    if word2 in thesaurus[word1] or word1 in thesaurus[word2]:
        return True
    return False

def add(thesaurus, similars, word):
    if similars == []:
        return thesaurus[word].copy()
    new_similars = []
    for sentence in similars:
        for synonym in thesaurus[word]:
            new_similars.append(sentence + ' ' + synonym)
    return new_similars

def get_similars(thesaurus, text):
    words = [clean(word) for word in text.strip().split(' ')]
    similars = []
    if len(words) > 3:
        print("Too many words...clipping to first 3")
    for word in words[:3]:
        similars = add(thesaurus, similars, word)
    return similars

def run(thesaurus):
    while True:
        text = input("Enter sentence or press q to quit:\n")
        if text in ['q', 'Q']:
            break
        similars = get_similars(thesaurus, text)
        print()
        for index in range(0, len(similars), 10):
            print('\n'.join(similars[index:index+10]))
            input()

def main():
    thesaurus = setup()
    run(thesaurus)

if __name__ == '__main__':
    main()
