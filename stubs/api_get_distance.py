import requests

from stubs.pickle_file import *

API_KEY = 'AIzaSyCly-jDt7xNsCxEve0CJq5iR2wjhd9vvxE'


def get_distance(city1: str, city2: str) -> int:
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "units": "metric",
        "origins": f"{city1}, BC, Canada",
        "destinations": f"{city2}, BC, Canada",
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    distance = data['rows'][0]['elements'][0]['distance']['value']
    return int(distance)


def pull_data(cities_: list, cost: dict) -> dict:
    for i, city1 in enumerate(cities_):
        for j, city2 in enumerate(cities_):
            if i < j:
                distance = get_distance(city1, city2)
                cost[(city1, city2)] = distance
                cost[(city2, city1)] = distance

    dump_data(cost)
    return cost


def generate_connectivity_map(cities: list) -> dict:
    """

    :param cities:
    :return:
    """
    CONNECTIVITY_MAP = {}
    for city in cities:
        CONNECTIVITY_MAP[city] = {}
        for other_city in cities:
            if city == other_city:
                continue
            CONNECTIVITY_MAP[city][other_city] = get_distance(city, other_city)
    return CONNECTIVITY_MAP


if __name__ == '__main__':
    cities = ["Vancouver", "North Vancouver", "West Vancouver", "Burnaby", "Coquitlam", "Delta", "Richmond", "Surrey",
              "Langley", "Abbotsford", "Mission", "Chilliwack", "New Westminster"]
    connectivity_map = generate_connectivity_map(cities)
    dump_data(connectivity_map)
