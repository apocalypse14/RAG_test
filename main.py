# This is a sample Python script.
from os import environ

from dotenv import load_dotenv


# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    load_dotenv()
    print(environ["OPENAI_API_KEY"])
    print(2**3)

