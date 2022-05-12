from bs4 import BeautifulSoup
import requests
import pandas as pd

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

cpi_dict = {}

try:
    source3 = requests.get('https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1913-')
    source3.raise_for_status()
    soup3 = BeautifulSoup(source3.text, 'html.parser')

    cpis = soup3.find('tbody').find_all('tr')[1:]

    for cpi in cpis:
        columns = cpi.find_all('td')
        cpi_dict[''.join([i for i in columns[0].div.text if i.isnumeric()])] = float(columns[1].div.text)
except:
    print('whoops')

def export_data(data):
    df = pd.DataFrame(data, columns = ['Name', 'Month', 'Year', 'Rating', 'Budget', 'Box Office', 'CPI for Year', 'CPI for 2021', 'Budget Adjusted for Inflation', 'Box Office Adjusted for Inflation'])
    df.to_excel("Stats Data.xlsx")

try:
    source = requests.get('https://www.imdb.com/chart/top/')
    source.raise_for_status()

    soup = BeautifulSoup(source.text, 'html.parser')

    movies = soup.find('tbody', class_ = "lister-list").find_all('tr')

    data = []

    for movie in movies:
        try:
            name = movie.find('td', class_ = "titleColumn").a.text

            year = movie.find('td', class_ = "titleColumn").span.text.strip('()')

            rating = movie.find('td', class_ = "ratingColumn imdbRating").strong.text

            link = movie.find('a')['href']

            source2 = requests.get('https://www.imdb.com' + link)
            source2.raise_for_status()
            soup2 = BeautifulSoup(source2.text, 'html.parser')
            movie2 = soup2.find('section', cel_widget_id = 'StaticFeature_Details')

            month = months.index(movie2.find('a', class_ = 'ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link').get_text().split(' ')[0]) + 1

            budget = soup2.find('section', cel_widget_id = 'StaticFeature_BoxOffice').find('span', class_ = 'ipc-metadata-list-item__list-content-item').get_text()
            budget = ''.join([i for i in budget if i.isnumeric()])

            try:
                opening_weekend = soup2.find('section', cel_widget_id = 'StaticFeature_BoxOffice').find('li', {'data-testid' : 'title-boxoffice-openingweekenddomestic'}).find('span', class_ = 'ipc-metadata-list-item__list-content-item').get_text()
                opening_weekend = ''.join([i for i in opening_weekend if i.isnumeric()])
            except:
                opening_weekend = 0

            if int((cpi_dict['2021']/cpi_dict[str(year)]) * int(budget)) < 400000000 and opening_weekend != 0:
                data.append([name, month, int(year), float(rating), int(budget), int(opening_weekend), cpi_dict[str(year)], cpi_dict['2021'],int((cpi_dict['2021']/cpi_dict[str(year)]) * int(budget)), int((cpi_dict['2021']/cpi_dict[str(year)]) * int(opening_weekend))])
                print(len(data))
        except:
            continue
except Exception as e:
    print(e)

export_data(data)