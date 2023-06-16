import requests
from bs4 import BeautifulSoup
class Sales_items:
    def __init__(self, name, price,product_url):
        self.name = name
        self.price = price
        self.url=product_url

req=requests.get('https://www.sklavenitis.gr/sylloges/prosfores/')
base_url='https://www.sklavenitis.gr'
sales_list=[]
if req.status_code==200:
    soup=BeautifulSoup(req.content,'html.parser')
    current_page=soup.find('div',{'class':'page'})
    span_page=current_page.span.text.strip().partition('τα ')[2].partition('προϊόντα')[0]
    total_pages=int(span_page.replace('.',''))
    count=0
    for page in range(1,total_pages//25):
        if count>=1:
            break
        req=requests.get(f'https://www.sklavenitis.gr/sylloges/prosfores/?pg={page}')
        product_list_id=soup.find('div',{'id':'productList'})
        product_list=product_list_id.find('section',{'class':'productList list-items-container'})
        product_divs=product_list.find_all('div')
        for product_div in product_divs:
            try:
                price_div=product_div.find('div',{'class':'price'})
                price=price_div.text.strip()
                name_div=product_div.find('article')
                name=name_div.h4.a.text.strip()
                url=name_div.h4.a["href"]
                product_url=f'{base_url}{url}'
                if product_url and name and price:
                    item=Sales_items(name,price,product_url)
                    sales_list.append(item)
            except Exception as error:
                print(error)
        count+=1
for item in sales_list:
    print(f'Found {item.name} on sale for {item.price} at webpage: {item.url}')


