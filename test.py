import requests


def test():
    response = requests.post(
        url="http://localhost:8765/parse_pdf/The_Minimum_Wages_Act_1948.pdf"
    )
    print(response.json())

def test2():
    response = requests.get(
        url="http://localhost:8765/generate_summary/The_Minimum_Wages_Act_1948.pdf/Overtime",
    )
    print(response.json())

def test3():
    response = requests.get(
        url="http://localhost:8765/original_section/The_Minimum_Wages_Act_1948.pdf/Overtime",
    )
    print(response.json())

test2()

# import pickle

# with open("The_Minimum_Wages_Act_1948.pdf_sections.pkl", "rb") as f:
#     sections = pickle.load(f)
#     print(sections.keys())
