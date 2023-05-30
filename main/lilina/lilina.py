print("Hello cats!")
print("My name is Lilina")

Hello="fuck"
print(Hello)
print('hello')
love="tsib"
print(f"I {love} my Dudu very much.")
shopping_list=["chocolate", "cheesecake", "mozzarella", "kinder bueno", "choco milk"]
print("Bibi buy me everything on the list. Thank you.")
for shopping_item in shopping_list:
    print(f"BiBi buy me {shopping_item}")

print("lesson 2")
to_do_today=["meeting", "shopping", "eating"]
week_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
for day in week_days:
    if day is 'Friday':
        print(f"On {day} we did a {to_do_today[0]}, then we went {to_do_today[1]}, and finally we went {to_do_today[-1]}")



####Homework####
name_list=['Alex','Katerina','Paul','Bob','Alex','Katerina','Baggelis','Alexandros','Alexander','Alexander','Alex','Bob','Anna','John','Jon','Alex','Alec','Jons','John']
#### Using .append like list.append(name) create a list that contains only the names that are excactly Alex
####Print the length of the list you created either by using the function: len , like : len(list) AND by using a counter inside a for loop
###like : for ...  counter+=1 (it adds one to the value of the counter everytime it loops)
###WARNING both when creating the list and the counter you might have to initialize them outside the loop function like :
### counter=0 .... for .... etc and alex_list=[] (this creates an empty list that is ready to be filled)

alex_list=[]
counter=0
for name in name_list:
    if name is "Alex":
        alex_list.append(name)
        counter+=1
print(len(alex_list))