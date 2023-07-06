seafood="kalamarakia"
seafood=seafood.replace("akia",'').replace("a","_")
print(seafood)

command="download"
command=command.replace("down","up").replace("up", "up-")
print(command)

sentence="H MARINA VARIETAI TO MATHIMA"
new_sentence=sentence.partition('VARIETAI')
new_sentence=sentence.partition('VARIETAI')[0]
new_sentence=sentence.partition('VARIETAI')[1]
new_sentence=sentence.partition('VARIETAI')[2]
print(new_sentence)
list_length=range(len(sentence.partition("VARIETAI")))
print(list_length)
print(sentence.partition('VARIETAI')[0],sentence.partition('VARIETAI')[1], sentence.partition('VARIETAI')[2])
for length in list_length:
    print(sentence.partition('VARIETAI')[length])

