print("This is my revision!")
flavour="caramel"
beverage="coffee"
print(f"I would like a {flavour} flavoured {beverage}, please.")

coffee_flavours=["vanilla", "caramel", "chocolate", "toffee", "brownies"]
for coffee_flavour in coffee_flavours:
    print(f"I would like a {coffee_flavour} flavoured {beverage}, please.")

beverages=["coffee", "whiskey", "water"]
for beverage in beverages:
    if beverage=="coffee":
        for coffee in coffee_flavours:
            print(f"I would like a {coffee} flavoured {beverage}, please.")
    elif beverage=="water":
        print(f"{coffee_flavours[0]}")
    else:
        print(f"I am sorry but {beverage} is not available")

